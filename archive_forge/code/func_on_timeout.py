from collections import Counter
from threading import Timer
import logging
import inspect
from ..core import MachineError, listify, State
@on_timeout.setter
def on_timeout(self, value):
    """ Listifies passed values and assigns them to on_timeout."""
    self._on_timeout = listify(value)