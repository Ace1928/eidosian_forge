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
def selection_result(self):
    if not self.selected_task:
        return

    def alert_callback(my, mx, xs):
        y = count(xs)
        task = self.state.tasks[self.selected_task]
        result = getattr(task, 'result', None) or getattr(task, 'exception', None)
        for line in wrap(result or '', mx - 2):
            self.win.addstr(next(y), 3, line)
    return self.alert(alert_callback, f'Task Result for {self.selected_task}')