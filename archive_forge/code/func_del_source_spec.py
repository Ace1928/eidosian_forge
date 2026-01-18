import os.path
import re
import sys
import shutil
from decimal import Decimal
from pyomo.common.dependencies import attempt_import
from pyomo.dataportal import TableData
from pyomo.dataportal.factory import DataManagerFactory
def del_source_spec(self, name):
    """
        Remove an ODBC data source specification from the
        configuration.
        """
    if name in self.source_specs:
        del self.source_specs[name]