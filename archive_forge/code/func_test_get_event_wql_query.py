from unittest import mock
import ddt
from oslo_utils import units
from os_win import constants
from os_win import exceptions
from os_win.tests.unit import test_base
from os_win.utils import _wqlutils
from os_win.utils.network import networkutils
def test_get_event_wql_query(self):
    expected = "SELECT * FROM %(event_type)s WITHIN %(timeframe)s WHERE TargetInstance ISA '%(class)s' AND %(like)s" % {'class': 'FakeClass', 'event_type': self.netutils.EVENT_TYPE_CREATE, 'like': "TargetInstance.foo LIKE 'bar%'", 'timeframe': 2}
    query = self.netutils._get_event_wql_query('FakeClass', self.netutils.EVENT_TYPE_CREATE, like=dict(foo='bar'))
    self.assertEqual(expected, query)