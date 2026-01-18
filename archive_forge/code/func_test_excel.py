import runpy
import sys
import os
from os.path import abspath, dirname
import pyomo.common.unittest as unittest
from pyomo.common.dependencies import pyutilib, pyutilib_available
@unittest.skipIf(not (_xlrd or _openpyxl), 'Cannot read excel file.')
@unittest.skipIf(not (_win32com and _excel_available and pyutilib_available), 'Cannot read excel file.')
def test_excel(self):
    self.driver('excel')