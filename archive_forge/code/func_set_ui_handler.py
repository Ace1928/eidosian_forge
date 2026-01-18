import contextlib
import logging
import threading
from threading import local as thread_local
from threading import Thread
import traceback
from types import MethodType
import weakref
import sys
from .constants import ComparisonMode, TraitKind
from .trait_base import Uninitialized
from .trait_errors import TraitNotificationError
def set_ui_handler(handler):
    """ Sets up the user interface thread handler.
    """
    global ui_handler, ui_thread
    ui_handler = handler
    ui_thread = threading.current_thread().ident