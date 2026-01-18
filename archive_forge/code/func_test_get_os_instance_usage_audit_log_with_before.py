import datetime
from oslo_utils import timeutils
from novaclient.tests.functional import base
def test_get_os_instance_usage_audit_log_with_before(self):
    expected = {'hosts_not_run': '[]', 'log': '{}', 'num_hosts': '0', 'num_hosts_done': '0', 'num_hosts_not_run': '0', 'num_hosts_running': '0', 'overall_status': 'ALL hosts done. 0 errors.', 'total_errors': '0', 'total_instances': '0', 'period_beginning': '2016-11-01 00:00:00', 'period_ending': '2016-12-01 00:00:00'}
    output = self.nova('instance-usage-audit-log --before "2016-12-10 13:59:59.999999"')
    for key in expected.keys():
        self.assertEqual(expected[key], self._get_value_from_the_table(output, key))