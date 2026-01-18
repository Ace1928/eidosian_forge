import copy
from unittest import mock
from osc_lib.tests import utils as osctestutils
from osc_lib import utils as oscutils
from ironicclient import exc
from ironicclient.osc.v1 import baremetal_port
from ironicclient.tests.unit.osc.v1 import fakes as baremetal_fakes
def test_baremetal_port_create_llc_warning_some_deprecated(self):
    self._test_baremetal_port_create_llc_warning(additional_args=['-l', 'port_id=eth0', '--local-link-connection', 'switch_id=aa:bb:cc:dd:ee:ff'], additional_verify_items=[('local_link_connection_deprecated', ['port_id=eth0']), ('local_link_connection', ['switch_id=aa:bb:cc:dd:ee:ff'])])