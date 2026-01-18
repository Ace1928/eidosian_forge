import inspect
import logging
import operator
import types
def is_same_callback(callback1, callback2):
    """Returns if the two callbacks are the same."""
    if callback1 is callback2:
        return True
    if callback1 == callback2:
        return True
    return False