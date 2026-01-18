import os
import pyomo.common.unittest as unittest
def test_init_empty_data(self):
    config = ODBCConfig()
    self.assertEqual({}, config.sources)
    self.assertEqual({}, config.source_specs)
    self.assertEqual({}, config.odbc_info)