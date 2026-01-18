import os
import pyomo.common.unittest as unittest
def test_del_source(self):
    config = ODBCConfig(data=self.simple_data)
    config.del_source('testdb')
    self.assertEqual({}, config.sources)