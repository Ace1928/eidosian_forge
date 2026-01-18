from __future__ import absolute_import, division, print_function
import re
import time
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
import ansible_collections.netapp.ontap.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.ontap.plugins.module_utils.netapp_module import NetAppModule
from ansible_collections.netapp.ontap.plugins.module_utils.netapp import OntapRestAPI
from ansible_collections.netapp.ontap.plugins.module_utils import rest_generic
import ansible_collections.netapp.ontap.plugins.module_utils.rest_response_helpers as rrh
def get_disks_to_add(self, aggr_name, disks, mirror_disks):
    """
        Get list of disks used by the aggregate, as primary and mirror.
        Report error if:
          the plexes in use cannot be matched with user inputs (we expect some overlap)
          the user request requires some disks to be removed (not supported)
        : return: a tuple of two lists of disks: disks_to_add, mirror_disks_to_add
        """
    disks_in_use = self.get_aggr_disks(aggr_name)
    plex_disks = {}
    for disk_name, plex_name in disks_in_use:
        plex_disks.setdefault(plex_name, []).append(disk_name)
    disks_plex, mirror_disks_plex = self.map_plex_to_primary_and_mirror(plex_disks, disks, mirror_disks)
    disks_to_remove = [disk for disk in plex_disks[disks_plex] if disk not in disks]
    if mirror_disks_plex:
        disks_to_remove.extend([disk for disk in plex_disks[mirror_disks_plex] if disk not in mirror_disks])
    if disks_to_remove:
        error = 'these disks cannot be removed: %s' % str(disks_to_remove)
        self.module.fail_json(msg='Error removing disks is not supported.  Aggregate %s: %s.  In use: %s' % (aggr_name, error, str(plex_disks)))
    disks_to_add = [disk for disk in disks if disk not in plex_disks[disks_plex]]
    mirror_disks_to_add = []
    if mirror_disks_plex:
        mirror_disks_to_add = [disk for disk in mirror_disks if disk not in plex_disks[mirror_disks_plex]]
    if mirror_disks_to_add and (not disks_to_add):
        self.module.fail_json(msg='Error cannot add mirror disks %s without adding disks for aggregate %s.  In use: %s' % (str(mirror_disks_to_add), aggr_name, str(plex_disks)))
    if disks_to_add or mirror_disks_to_add:
        self.na_helper.changed = True
    return (disks_to_add, mirror_disks_to_add)