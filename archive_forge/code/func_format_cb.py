import reprlib
from _thread import get_ident
from . import format_helpers
def format_cb(callback):
    return format_helpers._format_callback_source(callback, ())