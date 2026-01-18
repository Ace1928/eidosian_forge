from __future__ import absolute_import, division, print_function
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
import ansible_collections.netapp.ontap.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.ontap.plugins.module_utils.netapp_module import NetAppModule
from ansible_collections.netapp.ontap.plugins.module_utils import rest_generic
from ansible_collections.netapp.ontap.plugins.module_utils import rest_vserver
def identify_new_snapmirror_policy_rules(self, current=None):
    """
        Identify new rules that should be added.
        :return: List of new rules to be added
                 e.g. [{'snapmirror_label': 'daily', 'keep': 7, 'prefix': '', 'schedule': ''}, ... ]
        """
    new_rules = []
    if 'snapmirror_label' in self.parameters:
        for snapmirror_label in self.parameters['snapmirror_label']:
            snapmirror_label = snapmirror_label.strip()
            snapmirror_label_index = self.parameters['snapmirror_label'].index(snapmirror_label)
            rule = dict({'snapmirror_label': snapmirror_label, 'keep': self.parameters['keep'][snapmirror_label_index]})
            if 'prefix' in self.parameters:
                rule['prefix'] = self.parameters['prefix'][snapmirror_label_index]
            else:
                rule['prefix'] = ''
            if 'schedule' in self.parameters:
                rule['schedule'] = self.parameters['schedule'][snapmirror_label_index]
            else:
                rule['schedule'] = ''
            if current is None or 'snapmirror_label' not in current or snapmirror_label not in current['snapmirror_label']:
                new_rules.append(rule)
    return new_rules