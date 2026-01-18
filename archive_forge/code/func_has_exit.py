import sys
import queue
import threading
from pyglet import app
from pyglet import clock
from pyglet import event
@has_exit.setter
def has_exit(self, value):
    self._has_exit_condition.acquire()
    self._has_exit = value
    self._has_exit_condition.notify()
    self._has_exit_condition.release()