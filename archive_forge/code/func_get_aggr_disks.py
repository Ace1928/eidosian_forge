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
def get_aggr_disks(self, name):
    """
        Fetch disks that are used for this aggregate.
        :param name: Name of the aggregate to be fetched
        :return:
            list of tuples (disk-name, plex-name)
            empty list if aggregate is not found
        """
    disks = []
    aggr_get = self.disk_get_iter(name)
    if aggr_get and aggr_get.get_child_by_name('num-records') and (int(aggr_get.get_child_content('num-records')) >= 1):
        attr = aggr_get.get_child_by_name('attributes-list')
        disks = [(disk_info.get_child_content('disk-name'), disk_info.get_child_by_name('disk-raid-info').get_child_by_name('disk-aggregate-info').get_child_content('plex-name')) for disk_info in attr.get_children()]
    return disks