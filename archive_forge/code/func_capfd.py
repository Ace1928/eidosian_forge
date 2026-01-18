import os
import time
import sys
from io import StringIO, BytesIO
from pyomo.common.log import LoggingIntercept
import pyomo.common.unittest as unittest
from pyomo.common.tempfiles import TempfileManager
import pyomo.common.tee as tee
@unittest.pytest.fixture(autouse=True)
def capfd(self, capfd):
    """
        Reimplementation needed for use in unittest.TestCase subclasses
        """
    self.capfd = capfd