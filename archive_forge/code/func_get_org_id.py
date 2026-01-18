from __future__ import absolute_import, division, print_function
import time
import re
from ansible.module_utils.basic import json, env_fallback
from ansible.module_utils.common.dict_transformations import camel_dict_to_snake_dict, snake_dict_to_camel_dict, recursive_diff
from ansible.module_utils.urls import fetch_url
from ansible.module_utils.six.moves.urllib.parse import urlencode
from ansible.module_utils._text import to_native
def get_org_id(self, org_name):
    """Returns an organization id based on organization name, only if unique.

        If org_id is specified as parameter, return that instead of a lookup.
        """
    orgs = self.get_orgs()
    if self.params['org_id'] is not None:
        if self.is_org_valid(orgs, org_id=self.params['org_id']) is True:
            return self.params['org_id']
    org_count = self.is_org_valid(orgs, org_name=org_name)
    if org_count == 0:
        self.fail_json(msg='There are no organizations with the name {org_name}'.format(org_name=org_name))
    if org_count > 1:
        self.fail_json(msg='There are multiple organizations with the name {org_name}'.format(org_name=org_name))
    elif org_count == 1:
        for i in orgs:
            if org_name == i['name']:
                return str(i['id'])