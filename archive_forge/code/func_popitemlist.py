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
def popitemlist(self):
    try:
        key, buckets = dict.popitem(self)
    except KeyError as e:
        raise exceptions.BadRequestKeyError(e.args[0]) from None
    for bucket in buckets:
        bucket.unlink(self)
    return (key, [x.value for x in buckets])