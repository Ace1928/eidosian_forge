import contextlib
import threading
def get_config_under():
    try:
        value = _config.under
    except AttributeError:
        value = _config.under = None
    return value