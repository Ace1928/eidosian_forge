import functools
import logging
from sqlalchemy.ext.declarative import declarative_base
def sql_function(func):
    """
    Decorator for wrapping the given function in order to manipulate (CRUD)
    the records safely.

    For the adding/updating/deleting records function, this decorator
    invokes "Session.commit()" after the given function.
    If any exception while modifying records raised, this decorator invokes
    "Session.rollbacks()".
    """

    @functools.wraps(func)
    def _wrapper(session, *args, **kwargs):
        ret = None
        try:
            ret = func(session, *args, **kwargs)
            if session.dirty:
                session.commit()
        except Exception as e:
            LOG.error('Error in %s: %s', func.__name__, e)
            if session.dirty:
                LOG.error('Do rolling back %s table', session.dirty[0].__tablename__)
                session.rollback()
        return ret
    return _wrapper