from __future__ import annotations
import collections
import re
import typing
from typing import Any
from typing import Dict
from typing import Optional
import warnings
import weakref
from . import config
from .util import decorator
from .util import gc_collect
from .. import event
from .. import pool
from ..util import await_only
from ..util.typing import Literal
def after_test_outside_fixtures(self, test):
    if not config.bootstrapped_as_sqlalchemy:
        return
    if test.__class__.__leave_connections_for_teardown__:
        return
    self.checkin_all()
    from . import provision
    with config.db.connect() as conn:
        provision.prepare_for_drop_tables(conn.engine.url, conn)