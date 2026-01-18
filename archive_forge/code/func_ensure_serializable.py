import datetime
import numbers
import sys
from base64 import b64decode as base64decode
from base64 import b64encode as base64encode
from functools import partial
from inspect import getmro
from itertools import takewhile
from kombu.utils.encoding import bytes_to_str, safe_repr, str_to_bytes
def ensure_serializable(items, encoder):
    """Ensure items will serialize.

    For a given list of arbitrary objects, return the object
    or a string representation, safe for serialization.

    Arguments:
        items (Iterable[Any]): Objects to serialize.
        encoder (Callable): Callable function to serialize with.
    """
    safe_exc_args = []
    for arg in items:
        try:
            encoder(arg)
            safe_exc_args.append(arg)
        except Exception:
            safe_exc_args.append(safe_repr(arg))
    return tuple(safe_exc_args)