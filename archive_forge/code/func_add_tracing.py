import contextlib
import logging as log
from oslo_utils import reflection
from osprofiler import profiler
def add_tracing(sqlalchemy, engine, name, hide_result=True):
    """Add tracing to all sqlalchemy calls."""
    if not _DISABLED:
        sqlalchemy.event.listen(engine, 'before_cursor_execute', _before_cursor_execute(name))
        sqlalchemy.event.listen(engine, 'after_cursor_execute', _after_cursor_execute(hide_result=hide_result))
        sqlalchemy.event.listen(engine, 'handle_error', handle_error)