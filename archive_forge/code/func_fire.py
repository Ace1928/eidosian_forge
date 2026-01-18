from __future__ import unicode_literals
import inspect
import os
import signal
import sys
import threading
import weakref
from wcwidth import wcwidth
from six.moves import range
def fire(self):
    """ Alias for just calling the event. """
    self()