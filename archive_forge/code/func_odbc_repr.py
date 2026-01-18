import os.path
import re
import sys
import shutil
from decimal import Decimal
from pyomo.common.dependencies import attempt_import
from pyomo.dataportal import TableData
from pyomo.dataportal.factory import DataManagerFactory
def odbc_repr(self):
    """
        Get the full, odbc.ini-style representation of this
        ODBC configuration.
        """
    str = '[{0}]\n'.format(self.ODBC_DS_KEY)
    for name in self.sources:
        str += '{0} = {1}\n'.format(name, self.sources[name])
    for name in self.source_specs:
        str += '\n[{0}]\n'.format(name)
        for key in self.source_specs[name]:
            str += '{0} = {1}\n'.format(key, self.source_specs[name][key])
    if len(self.odbc_info) > 0:
        str += '\n[{0}]\n'.format(self.ODBC_INFO_KEY)
        for key in self.odbc_info:
            str += '{0} = {1}\n'.format(key, self.odbc_info[key])
    return str