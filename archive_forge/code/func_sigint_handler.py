import code
import greenlet
import logging
import signal
from curtsies.input import is_main_thread
def sigint_handler(self, *args):
    """SIGINT handler to use while code is running or request being
        fulfilled"""
    if greenlet.getcurrent() is self.code_context:
        logger.debug('sigint while running user code!')
        raise KeyboardInterrupt()
    else:
        logger.debug('sigint while fulfilling code request sigint handler running!')
        self.sigint_happened_in_main_context = True