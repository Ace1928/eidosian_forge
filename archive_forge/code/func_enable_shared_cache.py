import datetime
import time
import collections.abc
from _sqlite3 import *
def enable_shared_cache(enable):
    from _sqlite3 import enable_shared_cache as _old_enable_shared_cache
    import warnings
    msg = 'enable_shared_cache is deprecated and will be removed in Python 3.12. Shared cache is strongly discouraged by the SQLite 3 documentation. If shared cache must be used, open the database in URI mode usingthe cache=shared query parameter.'
    warnings.warn(msg, DeprecationWarning, stacklevel=2)
    return _old_enable_shared_cache(enable)