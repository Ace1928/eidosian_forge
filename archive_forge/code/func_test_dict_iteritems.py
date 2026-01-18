from heat.hacking import checks
from heat.tests import common
def test_dict_iteritems(self):
    self.assertEqual(1, len(list(checks.check_python3_no_iteritems('obj.iteritems()'))))
    self.assertEqual(0, len(list(checks.check_python3_no_iteritems('obj.items()'))))
    self.assertEqual(0, len(list(checks.check_python3_no_iteritems('obj.items()'))))