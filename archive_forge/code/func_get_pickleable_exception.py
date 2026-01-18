import datetime
import numbers
import sys
from base64 import b64decode as base64decode
from base64 import b64encode as base64encode
from functools import partial
from inspect import getmro
from itertools import takewhile
from kombu.utils.encoding import bytes_to_str, safe_repr, str_to_bytes
def get_pickleable_exception(exc):
    """Make sure exception is pickleable."""
    try:
        pickle.loads(pickle.dumps(exc))
    except Exception:
        pass
    else:
        return exc
    nearest = find_pickleable_exception(exc)
    if nearest:
        return nearest
    return UnpickleableExceptionWrapper.from_exception(exc)