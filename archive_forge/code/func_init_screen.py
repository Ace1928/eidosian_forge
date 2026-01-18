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
def init_screen(self):
    with self.lock:
        self.win = curses.initscr()
        self.win.nodelay(True)
        self.win.keypad(True)
        curses.start_color()
        curses.init_pair(1, self.foreground, self.background)
        curses.init_pair(2, curses.COLOR_RED, self.background)
        curses.init_pair(3, curses.COLOR_GREEN, self.background)
        curses.init_pair(4, curses.COLOR_MAGENTA, self.background)
        curses.init_pair(5, curses.COLOR_BLUE, self.background)
        curses.init_pair(6, curses.COLOR_YELLOW, self.foreground)
        self.state_colors = {states.SUCCESS: curses.color_pair(3), states.REVOKED: curses.color_pair(4), states.STARTED: curses.color_pair(6)}
        for state in states.EXCEPTION_STATES:
            self.state_colors[state] = curses.color_pair(2)
        curses.cbreak()