import itertools
import os
import struct
from copy import copy
from io import BytesIO
from typing import Any, Tuple
from zlib import adler32
from ..lazy_import import lazy_import
import fastbencode as bencode
from breezy import (
from breezy.bzr import (
from .. import errors
from .. import graph as _mod_graph
from .. import osutils
from .. import transport as _mod_transport
from ..registry import Registry
from ..textmerge import TextMerge
from . import index
def plan_merge(self, ver_a, ver_b, base=None):
    """See VersionedFile.plan_merge"""
    from ..merge import _PlanMerge
    if base is None:
        return _PlanMerge(ver_a, ver_b, self, (self._file_id,)).plan_merge()
    old_plan = list(_PlanMerge(ver_a, base, self, (self._file_id,)).plan_merge())
    new_plan = list(_PlanMerge(ver_a, ver_b, self, (self._file_id,)).plan_merge())
    return _PlanMerge._subtract_plans(old_plan, new_plan)