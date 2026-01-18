import collections
import hashlib
from functools import wraps
import flask
from .dependencies import (
from .exceptions import (
from ._grouping import (
from ._utils import (
from . import _validate
from .long_callback.managers import BaseLongCallbackManager
from ._callback_context import context_value
@staticmethod
def is_no_update(obj):
    return isinstance(obj, NoUpdate) or (isinstance(obj, dict) and obj == {'_dash_no_update': '_dash_no_update'})