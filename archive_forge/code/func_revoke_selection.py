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
def revoke_selection(self):
    if not self.selected_task:
        return curses.beep()
    reply = self.app.control.revoke(self.selected_task, reply=True)
    self.alert_remote_control_reply(reply)