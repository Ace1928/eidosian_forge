from __future__ import absolute_import, division, print_function
from copy import deepcopy
from ansible.module_utils.six import iteritems
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.cfg.base import (
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.utils import (
from ansible_collections.vyos.vyos.plugins.module_utils.network.vyos.facts.facts import Facts
from ansible_collections.vyos.vyos.plugins.module_utils.network.vyos.utils.utils import (
def execute_module(self):
    """Execute the module

        :rtype: A dictionary
        :returns: The result from module execution
        """
    result = {'changed': False}
    warnings = list()
    commands = list()
    if self.state in self.ACTION_STATES:
        existing_ospfv2_facts = self.get_ospfv2_facts()
    else:
        existing_ospfv2_facts = {}
    if self.state in self.ACTION_STATES or self.state == 'rendered':
        commands.extend(self.set_config(existing_ospfv2_facts))
    if commands and self.state in self.ACTION_STATES:
        if not self._module.check_mode:
            self._connection.edit_config(commands)
        result['changed'] = True
    if self.state in self.ACTION_STATES:
        result['commands'] = commands
    if self.state in self.ACTION_STATES or self.state == 'gathered':
        changed_ospfv2_facts = self.get_ospfv2_facts()
    elif self.state == 'rendered':
        result['rendered'] = commands
    elif self.state == 'parsed':
        running_config = self._module.params['running_config']
        if not running_config:
            self._module.fail_json(msg='value of running_config parameter must not be empty for state parsed')
        result['parsed'] = self.get_ospfv2_facts(data=running_config)
    else:
        changed_ospfv2_facts = {}
    if self.state in self.ACTION_STATES:
        result['before'] = existing_ospfv2_facts
        if result['changed']:
            result['after'] = changed_ospfv2_facts
    elif self.state == 'gathered':
        result['gathered'] = changed_ospfv2_facts
    result['warnings'] = warnings
    return result