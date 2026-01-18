import contextlib
import threading
def get_config_over():
    try:
        value = _config.over
    except AttributeError:
        value = _config.over = None
    return value