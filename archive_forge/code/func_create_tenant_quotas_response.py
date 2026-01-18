from __future__ import (absolute_import, division, print_function)
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.general.plugins.module_utils.manageiq import ManageIQ, manageiq_argument_spec
@staticmethod
def create_tenant_quotas_response(tenant_quotas):
    """ Creates the ansible result object from a manageiq tenant_quotas entity

        Returns:
            a dict with the applied quotas, name and value
        """
    if not tenant_quotas:
        return {}
    result = {}
    for quota in tenant_quotas:
        if quota['unit'] == 'bytes':
            value = float(quota['value']) / (1024 * 1024 * 1024)
        else:
            value = quota['value']
        result[quota['name']] = value
    return result