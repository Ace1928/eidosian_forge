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
def reconnecting_engine(url=None, options=None):
    url = url or config.db.url
    dbapi = config.db.dialect.dbapi
    if not options:
        options = {}
    options['module'] = ReconnectFixture(dbapi)
    engine = testing_engine(url, options)
    _dispose = engine.dispose

    def dispose():
        engine.dialect.dbapi.shutdown()
        engine.dialect.dbapi.is_stopped = False
        _dispose()
    engine.test_shutdown = engine.dialect.dbapi.shutdown
    engine.test_restart = engine.dialect.dbapi.restart
    engine.dispose = dispose
    return engine