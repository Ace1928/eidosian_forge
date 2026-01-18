import contextlib
import threading
def get_config_invalid():
    try:
        value = _config.invalid
    except AttributeError:
        value = _config.invalid = None
    return value