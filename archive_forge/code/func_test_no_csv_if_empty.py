from pyomo.common.dependencies import pandas as pd, pandas_available
import pyomo.common.unittest as unittest
import os
import pyomo.contrib.parmest.parmest as parmest
import pyomo.contrib.parmest.scenariocreator as sc
import pyomo.environ as pyo
from pyomo.environ import SolverFactory
@unittest.skipIf(not uuid_available, 'The uuid module is not available')
def test_no_csv_if_empty(self):
    emptyset = sc.ScenarioSet('empty')
    tfile = uuid.uuid4().hex + '.csv'
    emptyset.write_csv(tfile)
    self.assertFalse(os.path.exists(tfile), 'ScenarioSet wrote csv in spite of empty set')