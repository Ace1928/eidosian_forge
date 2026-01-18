from heat.hacking import checks
from heat.tests import common
def test_dict_iterkeys(self):
    self.assertEqual(1, len(list(checks.check_python3_no_iterkeys('obj.iterkeys()'))))
    self.assertEqual(0, len(list(checks.check_python3_no_iterkeys('obj.keys()'))))
    self.assertEqual(0, len(list(checks.check_python3_no_iterkeys('obj.keys()'))))