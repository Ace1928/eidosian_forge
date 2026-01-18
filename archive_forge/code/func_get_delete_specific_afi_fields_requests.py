from __future__ import absolute_import, division, print_function
from copy import deepcopy
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.cfg.base import (
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.utils import (
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.sonic import (
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.facts.facts import Facts
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.utils.utils import (
def get_delete_specific_afi_fields_requests(self, want_afi, have_afi):
    """creates and builds list of requests for deleting some fields of dhcp snooping config for
        one ip family. Each field checked and deleted independently from each other depending on if
        it is specified in playbook and matches with current config"""
    sent_commands = {}
    requests = []
    if want_afi.get('enabled') is True and have_afi.get('enabled') is True:
        sent_commands.update({'enabled': want_afi.get('enabled')})
        requests.extend(self.get_delete_enabled_request(want_afi))
    if want_afi.get('verify_mac') is False and have_afi.get('verify_mac') is False:
        sent_commands.update({'verify_mac': want_afi.get('verify_mac')})
        requests.extend(self.get_delete_verify_mac_request(want_afi))
    if want_afi.get('vlans') is not None and have_afi.get('vlans') is not None and (have_afi.get('vlans') != []):
        to_delete_vlans = have_afi['vlans']
        if len(want_afi['vlans']) > 0:
            to_delete_vlans = list(set(have_afi.get('vlans', [])).intersection(set(want_afi.get('vlans', []))))
        to_delete = {'afi': want_afi['afi'], 'vlans': to_delete_vlans}
        if len(to_delete['vlans']):
            sent_commands.update({'vlans': deepcopy(to_delete_vlans)})
            requests.extend(self.get_delete_vlans_requests(to_delete))
    if want_afi.get('trusted') is not None and have_afi.get('trusted') is not None and (have_afi.get('trusted') != []):
        to_delete_trusted = have_afi['trusted']
        if len(want_afi['trusted']) > 0:
            to_delete_trusted = want_afi['trusted']
            for intf in list(to_delete_trusted):
                if intf not in have_afi['trusted']:
                    to_delete_trusted.remove(intf)
        to_delete = {'afi': want_afi['afi'], 'trusted': to_delete_trusted}
        if len(to_delete['trusted']):
            sent_commands.update({'trusted': deepcopy(to_delete_trusted)})
            requests.extend(self.get_delete_trusted_requests(to_delete))
    if want_afi.get('source_bindings') is not None and have_afi.get('source_bindings') is not None and (have_afi.get('source_bindings') != []):
        to_delete_bindings = have_afi['source_bindings']
        if len(want_afi['source_bindings']) > 0:
            to_delete_bindings = want_afi['source_bindings']
            existing_keys = [binding['mac_addr'] for binding in have_afi['source_bindings']]
            for binding in list(to_delete_bindings):
                if binding['mac_addr'] not in existing_keys:
                    to_delete_bindings.remove(binding)
        to_delete = {'afi': want_afi['afi'], 'source_bindings': to_delete_bindings}
        if len(to_delete['source_bindings']):
            sent_commands.update({'source_bindings': deepcopy(to_delete_bindings)})
            requests.extend(self.get_delete_specific_source_bindings_requests(to_delete))
    return (sent_commands, requests)