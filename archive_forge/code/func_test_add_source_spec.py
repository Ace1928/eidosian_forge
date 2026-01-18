import os
import pyomo.common.unittest as unittest
def test_add_source_spec(self):
    config = ODBCConfig()
    config.add_source('testdb', self.ACCESS_CONFIGSTR)
    config.add_source_spec('testdb', {'Database': 'testdb.mdb'})
    self.assertEqual({'testdb': {'Database': 'testdb.mdb'}}, config.source_specs)