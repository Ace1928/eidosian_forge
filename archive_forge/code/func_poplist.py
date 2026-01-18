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
def poplist(self, key):
    buckets = dict.pop(self, key, ())
    for bucket in buckets:
        bucket.unlink(self)
    return [x.value for x in buckets]