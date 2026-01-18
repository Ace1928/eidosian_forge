from __future__ import absolute_import, division, print_function
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
import ansible_collections.netapp.ontap.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.ontap.plugins.module_utils.netapp_module import NetAppModule
from ansible_collections.netapp.ontap.plugins.module_utils import rest_generic
from ansible_collections.netapp.ontap.plugins.module_utils import rest_vserver
def identify_snapmirror_policy_rules_with_schedule(self, rules=None):
    """
        Identify rules that are using a schedule or not. At least one
        non-schedule rule must be added to a policy before schedule rules
        are added.
        :return: List of rules with schedules and a list of rules without schedules
                 e.g. [{'snapmirror_label': 'daily', 'keep': 7, 'prefix': 'daily', 'schedule': 'daily'}, ... ],
                      [{'snapmirror_label': 'weekly', 'keep': 5, 'prefix': '', 'schedule': ''}, ... ]
        """
    schedule_rules = []
    non_schedule_rules = []
    if rules is not None:
        for rule in rules:
            if 'schedule' in rule:
                schedule_rules.append(rule)
            else:
                non_schedule_rules.append(rule)
    return (schedule_rules, non_schedule_rules)