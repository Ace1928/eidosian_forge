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
def test_crud(self):
    mac_addresses = ['FA:16:3E:00:00:01', 'FA:16:3E:00:00:02']
    for mac in mac_addresses:
        mac = netaddr.EUI(mac)
        self._add_row(id=uuidutils.generate_uuid(), mac=mac)
        obj = self._get_one(mac)
        self.assertEqual(mac, obj.mac)
        random_mac = netaddr.EUI(net.get_random_mac(['fe', '16', '3e', '00', '00', '00']))
        self._update_row(mac, random_mac)
        obj = self._get_one(random_mac)
        self.assertEqual(random_mac, obj.mac)
    objs = self._get_all()
    self.assertEqual(len(mac_addresses), len(objs))
    self._delete_rows()
    objs = self._get_all()
    self.assertEqual(0, len(objs))