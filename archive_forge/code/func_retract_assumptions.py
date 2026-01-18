import threading
import time
from abc import ABCMeta, abstractmethod
def retract_assumptions(self, retracted, debug=False):
    self._command.retract_assumptions(retracted, debug)
    self._result = None