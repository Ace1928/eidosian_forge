import errno
import os
import re
from ..lazy_import import lazy_import
from breezy import (
from .. import transport as _mod_transport
from ..conflicts import Conflict as BaseConflict
from ..conflicts import ConflictList as BaseConflictList
from . import rio
def register_types(*conflict_types):
    """Register a Conflict subclass for serialization purposes"""
    global ctype
    for conflict_type in conflict_types:
        ctype[conflict_type.typestring] = conflict_type