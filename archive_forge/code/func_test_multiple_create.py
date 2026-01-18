import abc
import netaddr
from oslo_db import exception
from oslo_db.sqlalchemy import enginefacade
from oslo_db.sqlalchemy import test_fixtures
from oslo_utils import timeutils
from oslo_utils import uuidutils
import sqlalchemy as sa
from neutron_lib import context
from neutron_lib.db import sqlalchemytypes
from neutron_lib.tests import _base as test_base
from neutron_lib.tests import tools
from neutron_lib.utils import net
def test_multiple_create(self):
    ip_addresses = [{'name': 'fake_id1', 'ip': '10.0.0.5'}, {'name': 'fake_id2', 'ip': '10.0.0.1'}, {'name': 'fake_id3', 'ip': '2210::ffff:ffff:ffff:ffff'}, {'name': 'fake_id4', 'ip': '2120::ffff:ffff:ffff:ffff'}]
    self._test_multiple_create(ip_addresses)