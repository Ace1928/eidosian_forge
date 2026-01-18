from __future__ import absolute_import, division, print_function
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
import ansible_collections.netapp.ontap.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.ontap.plugins.module_utils.netapp_module import NetAppModule
from ansible_collections.netapp.ontap.plugins.module_utils.netapp import OntapRestAPI
from ansible_collections.netapp.ontap.plugins.module_utils import rest_generic
def create_snapshot_policy_rest(self):
    """
        create snapshot policy with rest API.
        """
    if not self.use_rest:
        return self.create_snapshot_policy()
    body = {'name': self.parameters.get('name'), 'enabled': self.parameters.get('enabled'), 'copies': []}
    if self.parameters.get('vserver'):
        body['svm.name'] = self.parameters['vserver']
    if 'comment' in self.parameters:
        body['comment'] = self.parameters['comment']
    if 'snapmirror_label' in self.parameters:
        snapmirror_labels = self.parameters['snapmirror_label']
    else:
        snapmirror_labels = [None] * len(self.parameters['schedule'])
    if 'prefix' in self.parameters:
        prefixes = self.parameters['prefix']
    else:
        prefixes = [None] * len(self.parameters['schedule'])
    for schedule, prefix, count, snapmirror_label in zip(self.parameters['schedule'], prefixes, self.parameters['count'], snapmirror_labels):
        copy = {'schedule': {'name': self.safe_strip(schedule)}, 'count': count}
        snapmirror_label = self.safe_strip(snapmirror_label)
        if snapmirror_label:
            copy['snapmirror_label'] = snapmirror_label
        prefix = self.safe_strip(prefix)
        if prefix:
            copy['prefix'] = prefix
        body['copies'].append(copy)
    api = 'storage/snapshot-policies'
    dummy, error = rest_generic.post_async(self.rest_api, api, body)
    if error is not None:
        self.module.fail_json(msg='Error on creating snapshot policy: %s' % error)