import os
import pyomo.common.unittest as unittest
def test_add_source(self):
    config = ODBCConfig()
    config.add_source('testdb', self.ACCESS_CONFIGSTR)
    self.assertEqual({'testdb': self.ACCESS_CONFIGSTR}, config.sources)
    self.assertEqual({}, config.source_specs)
    self.assertEqual({}, config.odbc_info)