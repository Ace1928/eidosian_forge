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
def selection_info(self):
    if not self.selected_task:
        return

    def alert_callback(mx, my, xs):
        my, mx = self.win.getmaxyx()
        y = count(xs)
        task = self.state.tasks[self.selected_task]
        info = task.info(extra=['state'])
        infoitems = [('args', info.pop('args', None)), ('kwargs', info.pop('kwargs', None))] + list(info.items())
        for key, value in infoitems:
            if key is None:
                continue
            value = str(value)
            curline = next(y)
            keys = key + ': '
            self.win.addstr(curline, 3, keys, curses.A_BOLD)
            wrapped = wrap(value, mx - 2)
            if len(wrapped) == 1:
                self.win.addstr(curline, len(keys) + 3, abbr(wrapped[0], self.screen_width - (len(keys) + 3)))
            else:
                for subline in wrapped:
                    nexty = next(y)
                    if nexty >= my - 1:
                        subline = ' ' * 4 + '[...]'
                    self.win.addstr(nexty, 3, abbr(' ' * 4 + subline, self.screen_width - 4), curses.A_NORMAL)
    return self.alert(alert_callback, f'Task details for {self.selected_task}')