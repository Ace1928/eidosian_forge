from __future__ import absolute_import, division, print_function
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
import ansible_collections.netapp.ontap.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.ontap.plugins.module_utils.netapp_module import NetAppModule
from ansible_collections.netapp.ontap.plugins.module_utils import rest_generic
def cgcreate(self):
    """
        Calls cg-start and cg-commit (when cg-start succeeds)
        """
    started = self.cg_start()
    if started:
        if self.cgid is not None:
            self.cg_commit()
        else:
            self.module.fail_json(msg='Error fetching CG ID for CG commit %s' % self.parameters['snapshot'], exception=traceback.format_exc())
    return started