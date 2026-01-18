from __future__ import absolute_import, division, print_function
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
import ansible_collections.netapp.ontap.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.ontap.plugins.module_utils.netapp_module import NetAppModule
from ansible_collections.netapp.ontap.plugins.module_utils import rest_generic
from ansible_collections.netapp.ontap.plugins.module_utils import rest_vserver
def identify_obsolete_snapmirror_policy_rules(self, current=None):
    """
        Identify existing rules that should be deleted
        :return: List of rules to be deleted
                 e.g. [{'snapmirror_label': 'daily', 'keep': 7, 'prefix': '', 'schedule': ''}, ... ]
        """
    obsolete_rules = []
    if 'snapmirror_label' in self.parameters and current is not None and ('snapmirror_label' in current):
        for snapmirror_label in current['snapmirror_label']:
            snapmirror_label = snapmirror_label.strip()
            if snapmirror_label not in [item.strip() for item in self.parameters['snapmirror_label']]:
                current_snapmirror_label_index = current['snapmirror_label'].index(snapmirror_label)
                rule = dict({'snapmirror_label': snapmirror_label, 'keep': current['keep'][current_snapmirror_label_index], 'prefix': current['prefix'][current_snapmirror_label_index], 'schedule': current['schedule'][current_snapmirror_label_index]})
                obsolete_rules.append(rule)
    return obsolete_rules