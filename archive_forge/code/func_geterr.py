import contextlib
import threading
def geterr():
    """
    TODO(hvy): Write docs.
    """
    return dict(divide=get_config_divide(), over=get_config_over(), under=get_config_under(), invalid=get_config_invalid(), linalg=get_config_linalg(), fallback_mode=get_config_fallback_mode())