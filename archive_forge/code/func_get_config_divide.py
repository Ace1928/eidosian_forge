import contextlib
import threading
def get_config_divide():
    try:
        value = _config.divide
    except AttributeError:
        value = _config.divide = None
    return value