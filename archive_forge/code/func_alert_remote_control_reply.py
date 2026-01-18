import curses
import sys
import threading
from datetime import datetime
from itertools import count
from math import ceil
from textwrap import wrap
from time import time
from celery import VERSION_BANNER, states
from celery.app import app_or_default
from celery.utils.text import abbr, abbrtask
def alert_remote_control_reply(self, reply):

    def callback(my, mx, xs):
        y = count(xs)
        if not reply:
            self.win.addstr(next(y), 3, 'No replies received in 1s deadline.', curses.A_BOLD + curses.color_pair(2))
            return
        for subreply in reply:
            curline = next(y)
            host, response = next(subreply.items())
            host = f'{host}: '
            self.win.addstr(curline, 3, host, curses.A_BOLD)
            attr = curses.A_NORMAL
            text = ''
            if 'error' in response:
                text = response['error']
                attr |= curses.color_pair(2)
            elif 'ok' in response:
                text = response['ok']
                attr |= curses.color_pair(3)
            self.win.addstr(curline, 3 + len(host), text, attr)
    return self.alert(callback, 'Remote Control Command Replies')