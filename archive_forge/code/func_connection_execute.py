from __future__ import annotations
import collections
import contextlib
import itertools
import re
from .. import event
from ..engine import url
from ..engine.default import DefaultDialect
from ..schema import BaseDDLElement
@event.listens_for(engine, 'before_execute')
def connection_execute(conn, clauseelement, multiparams, params, execution_options):
    orig[:] = (clauseelement, multiparams, params)