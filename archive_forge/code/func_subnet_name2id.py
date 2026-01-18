from neutronclient._i18n import _
from neutronclient.neutron import v2_0 as neutronv20
def subnet_name2id(self, endpoint):
    return neutronv20.find_resourceid_by_name_or_id(self.get_client(), 'subnet', endpoint)