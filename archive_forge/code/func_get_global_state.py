import time
import codecs
import sys
def get_global_state():
    if _global_state is None:
        return initialize()
    return _global_state