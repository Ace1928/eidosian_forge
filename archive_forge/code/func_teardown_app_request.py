from __future__ import annotations
import os
import typing as t
from collections import defaultdict
from functools import update_wrapper
from .. import typing as ft
from .scaffold import _endpoint_from_view_func
from .scaffold import _sentinel
from .scaffold import Scaffold
from .scaffold import setupmethod
@setupmethod
def teardown_app_request(self, f: T_teardown) -> T_teardown:
    """Like :meth:`teardown_request`, but after every request, not only those
        handled by the blueprint. Equivalent to :meth:`.Flask.teardown_request`.
        """
    self.record_once(lambda s: s.app.teardown_request_funcs.setdefault(None, []).append(f))
    return f