from __future__ import absolute_import, division, print_function
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.cfg.base import (
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.utils import (
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.facts.facts import Facts
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.utils.utils import (
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.sonic import to_request
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.sonic import (
from ansible.module_utils.connection import ConnectionError
import json
from ansible.module_utils._text import to_native
import traceback
def get_new_add_request(self, conf):
    url = 'data/openconfig-routing-policy:routing-policy/defined-sets/openconfig-bgp-policy:bgp-defined-sets/community-sets'
    method = 'PATCH'
    community_members = []
    community_action = ''
    if 'match' not in conf:
        conf['match'] = 'ANY'
    if conf['type'] == 'standard':
        for attr in self.standard_communities_map:
            if attr in conf and conf[attr]:
                community_members.append(self.standard_communities_map[attr])
        if 'members' in conf and conf['members'] and conf['members'].get('regex', []):
            for i in conf['members']['regex']:
                community_members.extend([str(i)])
        if not community_members:
            self._module.fail_json(msg='Cannot create standard community-list {0} without community attributes'.format(conf['name']))
    elif conf['type'] == 'expanded':
        if 'members' in conf and conf['members'] and conf['members'].get('regex', []):
            for i in conf['members']['regex']:
                community_members.extend(['REGEX:' + str(i)])
        if not community_members:
            self._module.fail_json(msg='Cannot create expanded community-list {0} without community attributes'.format(conf['name']))
    if conf['permit']:
        community_action = 'PERMIT'
    else:
        community_action = 'DENY'
    input_data = {'name': conf['name'], 'members_list': community_members, 'match': conf['match'].upper(), 'permit': community_action}
    payload_template = '\n                            {\n                                "openconfig-bgp-policy:community-sets": {\n                                    "community-set": [\n                                        {\n                                            "community-set-name": "{{name}}",\n                                            "config": {\n                                                "community-set-name": "{{name}}",\n                                                "community-member": [\n                                                    {% for member in members_list %}"{{member}}"{%- if not loop.last -%},{% endif %}{%endfor%}\n                                                ],\n                                                "openconfig-bgp-policy-ext:action": "{{permit}}",\n                                                "match-set-options": "{{match}}"\n                                            }\n                                        }\n                                    ]\n                                }\n                            }'
    env = jinja2.Environment(autoescape=False)
    t = env.from_string(payload_template)
    intended_payload = t.render(input_data)
    ret_payload = json.loads(intended_payload)
    request = {'path': url, 'method': method, 'data': ret_payload}
    return request