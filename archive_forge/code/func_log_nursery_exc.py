import builtins
import logging
import signal
import threading
import traceback
import warnings
import trio
def log_nursery_exc(exc):
    exc = '\n'.join(traceback.format_exception(type(exc), exc, exc.__traceback__))
    logging.error('An exception occurred in a global nursery task.\n%s', exc)