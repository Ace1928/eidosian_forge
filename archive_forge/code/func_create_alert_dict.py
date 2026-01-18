from __future__ import (absolute_import, division, print_function)
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.general.plugins.module_utils.manageiq import ManageIQ, manageiq_argument_spec
def create_alert_dict(self, params):
    """ Create a dict representing an alert
        """
    if params['expression_type'] == 'hash':
        self.validate_hash_expression(params['expression'])
        expression_type = 'hash_expression'
    else:
        expression_type = 'expression'
    alert = dict(description=params['description'], db=params['resource_type'], options=params['options'], enabled=params['enabled'])
    alert.update({expression_type: params['expression']})
    return alert