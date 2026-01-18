from heat.hacking import checks
from heat.tests import common
def test_dict_itervalues(self):
    self.assertEqual(1, len(list(checks.check_python3_no_itervalues('obj.itervalues()'))))
    self.assertEqual(0, len(list(checks.check_python3_no_itervalues('obj.values()'))))
    self.assertEqual(0, len(list(checks.check_python3_no_itervalues('obj.values()'))))