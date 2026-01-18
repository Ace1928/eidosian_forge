from __future__ import annotations
from collections.abc import MutableSet
from copy import deepcopy
from .. import exceptions
from .._internal import _missing
from .mixins import ImmutableDictMixin
from .mixins import ImmutableListMixin
from .mixins import ImmutableMultiDictMixin
from .mixins import UpdateDictMixin
from .. import http
def getlist(self, key, type=None):
    rv = []
    for d in self.dicts:
        rv.extend(d.getlist(key, type))
    return rv