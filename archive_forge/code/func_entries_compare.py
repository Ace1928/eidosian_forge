from __future__ import absolute_import, division, print_function
from ansible.module_utils.six import iteritems
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.rm_base.resource_module import (
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.utils import (
from ansible_collections.cisco.ios.plugins.module_utils.network.ios.facts.facts import Facts
from ansible_collections.cisco.ios.plugins.module_utils.network.ios.rm_templates.route_maps import (
def entries_compare(self, want, have):
    if want.get('entries'):
        cmd_len = len(self.commands)
        if have.get('entries'):
            for k, v in iteritems(want['entries']):
                have_entry = have['entries'].pop(k, {})
                if want['entries'][k] != have_entry:
                    if (self.state == 'replaced' or self.state == 'overridden') and have_entry.get('description') and (have_entry.get('description') != want['entries'][k].get('description')):
                        self.compare(parsers=['description'], want=dict(), have=have_entry)
                    self.compare(parsers=self.parsers, want=want['entries'][k], have=have_entry)
                    have_match = have_entry.get('match')
                    want_match = v.get('match')
                    if have_match and want_match:
                        self.list_type_compare('match', want=want_match, have=have_match)
                    elif not have_match and want_match:
                        self.list_type_compare('match', want=want_match, have=dict())
                    have_set = have_entry.get('set')
                    want_set = v.get('set')
                    if have_set and want_set:
                        self.list_type_compare('set', want=want_set, have=have_set)
                    elif not have_set and want_set:
                        self.list_type_compare('set', want=want_set, have=dict())
                if cmd_len != len(self.commands):
                    route_map_cmd = 'route-map {route_map}'.format(**want)
                    if want['entries'][k].get('action'):
                        route_map_cmd += ' {action}'.format(**want['entries'][k])
                    if want['entries'][k].get('sequence'):
                        route_map_cmd += ' {sequence}'.format(**want['entries'][k])
                    self.commands.insert(cmd_len, route_map_cmd)
                    cmd_len = len(self.commands)
        else:
            for k, v in iteritems(want['entries']):
                self.compare(parsers=self.parsers, want=want['entries'][k], have=dict())
                want_match = v.get('match')
                if want_match:
                    self.list_type_compare('match', want=want_match, have=dict())
                want_set = v.get('set')
                if want_set:
                    self.list_type_compare('set', want=want_set, have=dict())
                if cmd_len != len(self.commands):
                    route_map_cmd = 'route-map {route_map}'.format(**want)
                    if want['entries'][k].get('action'):
                        route_map_cmd += ' {action}'.format(**want['entries'][k])
                    if want['entries'][k].get('sequence'):
                        route_map_cmd += ' {sequence}'.format(**want['entries'][k])
                    self.commands.insert(cmd_len, route_map_cmd)
                    cmd_len = len(self.commands)
    if (self.state == 'replaced' or self.state == 'overridden') and have.get('entries'):
        cmd_len = len(self.commands)
        for k, v in iteritems(have['entries']):
            route_map_cmd = 'no route-map {route_map}'.format(**have)
            if have['entries'][k].get('action'):
                route_map_cmd += ' {action}'.format(**have['entries'][k])
            if have['entries'][k].get('sequence'):
                route_map_cmd += ' {sequence}'.format(**have['entries'][k])
            self.commands.insert(cmd_len, route_map_cmd)