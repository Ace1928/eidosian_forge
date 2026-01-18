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
def get_aggr(self, name=None):
    """
        Fetch details if aggregate exists.
        :param name: Name of the aggregate to be fetched
        :return:
            Dictionary of current details if aggregate found
            None if aggregate is not found
        """
    if name is None:
        name = self.parameters.get('name')
    if self.use_rest:
        return self.get_aggr_rest(name)
    aggr_get = self.aggr_get_iter(name)
    if aggr_get and aggr_get.get_child_by_name('num-records') and (int(aggr_get.get_child_content('num-records')) >= 1):
        attr = aggr_get.get_child_by_name('attributes-list').get_child_by_name('aggr-attributes')
        current_aggr = {'service_state': attr.get_child_by_name('aggr-raid-attributes').get_child_content('state')}
        if attr.get_child_by_name('aggr-raid-attributes').get_child_content('disk-count'):
            current_aggr['disk_count'] = int(attr.get_child_by_name('aggr-raid-attributes').get_child_content('disk-count'))
        if attr.get_child_by_name('aggr-raid-attributes').get_child_content('encrypt-with-aggr-key'):
            current_aggr['encryption'] = attr.get_child_by_name('aggr-raid-attributes').get_child_content('encrypt-with-aggr-key') == 'true'
        snaplock_type = self.na_helper.safe_get(attr, ['aggr-snaplock-attributes', 'snaplock-type'])
        if snaplock_type:
            current_aggr['snaplock_type'] = snaplock_type
        return current_aggr
    return None