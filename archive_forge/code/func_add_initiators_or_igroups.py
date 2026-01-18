from __future__ import absolute_import, division, print_function
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
import ansible_collections.netapp.ontap.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.ontap.plugins.module_utils.netapp_module import NetAppModule
from ansible_collections.netapp.ontap.plugins.module_utils.netapp import OntapRestAPI
from ansible_collections.netapp.ontap.plugins.module_utils import rest_generic
def add_initiators_or_igroups(self, uuid, option, current_names):
    """
        Add the list of desired initiators to igroup unless they are already set
        :return: None
        """
    self.check_option_is_valid(option)
    if self.parameters.get(option) == [''] or self.parameters.get(option) is None:
        return
    names_to_add = [name for name in self.parameters[option] if name not in current_names]
    if self.use_rest and names_to_add:
        self.add_initiators_or_igroups_rest(uuid, option, names_to_add)
    else:
        for name in names_to_add:
            self.modify_initiator(name, 'igroup-add')