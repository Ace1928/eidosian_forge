from unittest import mock
import ddt
from manilaclient import api_versions
from manilaclient.common.apiclient import exceptions as client_exceptions
from manilaclient import exceptions
from manilaclient import extension
from manilaclient.tests.unit import utils
from manilaclient.tests.unit.v2 import fakes
from manilaclient.v2 import shares
@ddt.data(('id', 'b4991315-eb7d-43ec-979e-5715d4399827', True), ('id', 'b4991315-eb7d-43ec-979e-5715d4399827', False), ('path', 'fake_path', False), ('path', 'fake_path', True))
@ddt.unpack
def test_list_shares_index_with_export_location(self, filter_type, value, detailed):
    search_opts = {'export_location': value}
    cs.shares.list(detailed=detailed, search_opts=search_opts)
    if detailed:
        cs.assert_called('GET', '/shares/detail?export_location_' + filter_type + '=' + value + '&is_public=True')
    else:
        cs.assert_called('GET', '/shares?export_location_' + filter_type + '=' + value + '&is_public=True')