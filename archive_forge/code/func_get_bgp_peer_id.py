from neutronclient._i18n import _
from neutronclient.common import exceptions
from neutronclient.common import utils
from neutronclient.common import validators
from neutronclient.neutron import v2_0 as neutronv20
def get_bgp_peer_id(client, id_or_name):
    return neutronv20.find_resourceid_by_name_or_id(client, 'bgp_peer', id_or_name)