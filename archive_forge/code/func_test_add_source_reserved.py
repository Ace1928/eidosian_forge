import os
import pyomo.common.unittest as unittest
def test_add_source_reserved(self):
    config = ODBCConfig()
    with self.assertRaises(ODBCError):
        config.add_source('ODBC Data Sources', self.ACCESS_CONFIGSTR)
    with self.assertRaises(ODBCError):
        config.add_source('ODBC', self.ACCESS_CONFIGSTR)