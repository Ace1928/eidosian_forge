from collections import deque
from ..lazy_import import lazy_import
from breezy.bzr import (
from .. import errors, lazy_regex, osutils, trace
from .static_tuple import StaticTuple
def ensure_normalized_name(name):
    """Normalize name.

    :raises InvalidNormalization: When name is not normalized, and cannot be
        accessed on this platform by the normalized path.
    :return: The NFC normalised version of name.
    """
    norm_name, can_access = osutils.normalized_filename(name)
    if norm_name != name:
        if can_access:
            return norm_name
        else:
            raise errors.InvalidNormalization(name)
    return name