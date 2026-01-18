import code
import greenlet
import logging
import signal
from curtsies.input import is_main_thread
def load_code(self, source):
    """Prep code to be run"""
    assert self.source is None, "you shouldn't load code when some is already running"
    self.source = source
    self.code_context = None