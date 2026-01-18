from __future__ import annotations
import collections
import contextlib
import itertools
import re
from .. import event
from ..engine import url
from ..engine.default import DefaultDialect
from ..schema import BaseDDLElement
@event.listens_for(engine, 'after_cursor_execute')
def cursor_execute(conn, cursor, statement, parameters, context, executemany):
    if not context:
        return
    if asserter.accumulated and asserter.accumulated[-1].context is context:
        obs = asserter.accumulated[-1]
    else:
        obs = SQLExecuteObserved(context, orig[0], orig[1], orig[2])
        asserter.accumulated.append(obs)
    obs.statements.append(SQLCursorExecuteObserved(statement, parameters, context, executemany))