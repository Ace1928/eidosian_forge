from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ..module_utils.cloudstack import (
def get_affinity_type(self):
    affinity_type = self.module.params.get('affinity_type')
    affinity_types = self.query_api('listAffinityGroupTypes')
    if affinity_types:
        if not affinity_type:
            return affinity_types['affinityGroupType'][0]['type']
        for a in affinity_types['affinityGroupType']:
            if a['type'] == affinity_type:
                return a['type']
    self.module.fail_json(msg='affinity group type not found: %s' % affinity_type)