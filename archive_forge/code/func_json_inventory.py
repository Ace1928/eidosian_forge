from __future__ import (absolute_import, division, print_function)
from ansible.cli import CLI
import sys
import argparse
from ansible import constants as C
from ansible import context
from ansible.cli.arguments import option_helpers as opt_help
from ansible.errors import AnsibleError, AnsibleOptionsError
from ansible.module_utils.common.text.converters import to_bytes, to_native, to_text
from ansible.utils.vars import combine_vars
from ansible.utils.display import Display
from ansible.vars.plugins import get_vars_from_inventory_sources, get_vars_from_path
def json_inventory(self, top):
    seen_groups = set()

    def format_group(group, available_hosts):
        results = {}
        results[group.name] = {}
        if group.name != 'all':
            results[group.name]['hosts'] = [h.name for h in group.hosts if h.name in available_hosts]
        results[group.name]['children'] = []
        for subgroup in group.child_groups:
            results[group.name]['children'].append(subgroup.name)
            if subgroup.name not in seen_groups:
                results.update(format_group(subgroup, available_hosts))
                seen_groups.add(subgroup.name)
        if context.CLIARGS['export']:
            results[group.name]['vars'] = self._get_group_variables(group)
        self._remove_empty_keys(results[group.name])
        if not results[group.name]:
            del results[group.name]
        return results
    hosts = self.inventory.get_hosts(top.name)
    results = format_group(top, frozenset((h.name for h in hosts)))
    results['_meta'] = {'hostvars': {}}
    for host in hosts:
        hvars = self._get_host_variables(host)
        if hvars:
            results['_meta']['hostvars'][host.name] = hvars
    return results