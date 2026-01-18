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
def selection_traceback(self):
    if not self.selected_task:
        return curses.beep()
    task = self.state.tasks[self.selected_task]
    if task.state not in states.EXCEPTION_STATES:
        return curses.beep()

    def alert_callback(my, mx, xs):
        y = count(xs)
        for line in task.traceback.split('\n'):
            self.win.addstr(next(y), 3, line)
    return self.alert(alert_callback, f'Task Exception Traceback for {self.selected_task}')