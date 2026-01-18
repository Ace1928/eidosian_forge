import os.path
import re
import sys
import shutil
from decimal import Decimal
from pyomo.common.dependencies import attempt_import
from pyomo.dataportal import TableData
from pyomo.dataportal.factory import DataManagerFactory
def set_odbc_info(self, key, value):
    """
        Set an option for the ODBC handling specified in the
        configuration. An option consists of a key-value pair.
        Specifying an existing key will update the current value.
        """
    if key is None or value is None or len(key) == 0 or (len(value) == 0):
        raise ODBCError('An ODBC info pair must specify both a key and a value')
    self.odbc_info[str(key)] = str(value)