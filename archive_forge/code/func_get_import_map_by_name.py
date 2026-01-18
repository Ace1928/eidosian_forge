from os_ken.services.protocols.bgp.info_base.vrf import VrfRtImportMap
from os_ken.services.protocols.bgp.info_base.vrf4 import Vrf4NlriImportMap
from os_ken.services.protocols.bgp.info_base.vrf6 import Vrf6NlriImportMap
def get_import_map_by_name(self, name):
    return self._import_maps_by_name.get(name)