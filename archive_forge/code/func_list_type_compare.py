from __future__ import absolute_import, division, print_function
from ansible.module_utils.six import iteritems
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.rm_base.resource_module import (
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.utils import (
from ansible_collections.cisco.ios.plugins.module_utils.network.ios.facts.facts import Facts
from ansible_collections.cisco.ios.plugins.module_utils.network.ios.rm_templates.route_maps import (
def list_type_compare(self, compare_type, want, have):
    parsers = ['{0}'.format(compare_type), '{0}.ip'.format(compare_type), '{0}.ipv6'.format(compare_type)]
    for k, v in iteritems(want):
        have_v = have.pop(k, {})
        if v != have_v and k not in ['ip', 'ipv6', 'action', 'sequence', 'community']:
            if have_v:
                self.compare(parsers=parsers, want={compare_type: {k: v}}, have={compare_type: {k: have_v}})
            else:
                self.compare(parsers=parsers, want={compare_type: {k: v}}, have=dict())
        if k in ['community']:
            if have_v:
                if have_v != v:
                    if self.state == 'overridden' or self.state == 'replaced':
                        self.compare(parsers=parsers, want={}, have={compare_type: {k: have_v}})
                    elif self.state == 'merged':
                        for _key, _val in have_v.items():
                            if isinstance(_val, list):
                                v[_key].extend(_val)
                                v[_key] = list(set(v[_key]))
                                v[_key].sort()
                    self.compare(parsers=parsers, want={compare_type: {k: v}}, have={compare_type: {k: have_v}})
            else:
                self.compare(parsers=parsers, want={compare_type: {k: v}}, have=dict())
        if k in ['ip', 'ipv6']:
            for key, val in iteritems(v):
                have_val = have_v.pop(key, {})
                if val != have_val:
                    if have_val:
                        if self.state == 'overridden' or self.state == 'replaced':
                            self.compare(parsers=parsers, want=dict(), have={compare_type: {k: {key: have_val}}})
                        self.compare(parsers=parsers, want={compare_type: {k: {key: val}}}, have={compare_type: {k: {key: have_val}}})
                    else:
                        self.compare(parsers=parsers, want={compare_type: {k: {key: val}}}, have=dict())
            if (self.state == 'overridden' or self.state == 'replaced') and have_v:
                for key, val in iteritems(have_v):
                    self.compare(parsers=parsers, want=dict(), have={compare_type: {k: {key: val}}})
    if have and (self.state == 'replaced' or self.state == 'overridden'):
        for k, v in iteritems(have):
            if k in ['ip', 'ipv6']:
                for key, val in iteritems(v):
                    if key and val:
                        self.compare(parsers=parsers, want=dict(), have={compare_type: {k: {key: val}}})
            else:
                self.compare(parsers=parsers, want=dict(), have={compare_type: {k: v}})