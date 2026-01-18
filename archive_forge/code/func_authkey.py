import os
import sys
import signal
import itertools
import threading
from _weakrefset import WeakSet
@authkey.setter
def authkey(self, authkey):
    """
        Set authorization key of process
        """
    self._config['authkey'] = AuthenticationString(authkey)