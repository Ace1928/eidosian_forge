from __future__ import absolute_import, division, print_function
import re
from ansible_collections.netapp_eseries.santricity.plugins.module_utils.santricity import NetAppESeriesModule
from ansible.module_utils._text import to_native
def get_host_board_id(self, iface_ref):
    if self.get_host_board_id_cache is None:
        try:
            rc, iface_board_map_list = self.request('storage-systems/%s/graph/xpath-filter?query=/ioInterfaceHicMap' % self.ssid)
        except Exception as err:
            self.module.fail_json(msg='Failed to retrieve IO interface HIC mappings! Array Id [%s]. Error [%s].' % (self.ssid, to_native(err)))
        self.get_host_board_id_cache = dict()
        for iface_board_map in iface_board_map_list:
            self.get_host_board_id_cache.update({iface_board_map['interfaceRef']: iface_board_map['hostBoardRef']})
    return self.get_host_board_id_cache[iface_ref]