import contextlib
import threading
def get_config_fallback_mode():
    try:
        value = _config.fallback_mode
    except AttributeError:
        value = _config.fallback_mode = 'ignore'
    return value