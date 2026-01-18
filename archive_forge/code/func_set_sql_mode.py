import re
from . import cursors, _mysql
from ._exceptions import (
def set_sql_mode(self, sql_mode):
    """Set the connection sql_mode. See MySQL documentation for
        legal values."""
    if self._server_version < (4, 1):
        raise NotSupportedError('server is too old to set sql_mode')
    self.query("SET SESSION sql_mode='%s'" % sql_mode)
    self.store_result()