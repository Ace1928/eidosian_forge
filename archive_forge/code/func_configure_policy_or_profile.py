from __future__ import (absolute_import, division, print_function)
from base64 import b64encode
from email.utils import formatdate
import re
import json
import hashlib
from ansible.module_utils.six import iteritems
from ansible.module_utils.six.moves.urllib.parse import urlparse, urlencode
from ansible.module_utils.urls import fetch_url
from ansible.module_utils.basic import env_fallback
def configure_policy_or_profile(self, resource_path):
    organization_moid = None
    self.get_resource(resource_path='/organization/Organizations', query_params={'$filter': "Name eq '" + self.module.params['organization'] + "'", '$select': 'Moid'})
    if self.result['api_response'].get('Moid'):
        organization_moid = self.result['api_response']['Moid']
    self.result['api_response'] = {}
    filter_str = "Name eq '" + self.module.params['name'] + "'"
    filter_str += "and Organization.Moid eq '" + organization_moid + "'"
    self.get_resource(resource_path=resource_path, query_params={'$filter': filter_str, '$expand': 'Organization'})
    moid = None
    resource_values_match = False
    if self.result['api_response'].get('Moid'):
        moid = self.result['api_response']['Moid']
        if self.module.params['state'] == 'present':
            resource_values_match = compare_values(self.api_body, self.result['api_response'])
        else:
            self.delete_resource(moid=moid, resource_path=resource_path)
            moid = None
    if self.module.params['state'] == 'present' and (not resource_values_match):
        self.api_body.pop('Organization')
        if not moid:
            self.api_body['Organization'] = {'Moid': organization_moid}
        self.configure_resource(moid=moid, resource_path=resource_path, body=self.api_body, query_params={'$filter': filter_str})
        if self.result['api_response'].get('Moid'):
            moid = self.result['api_response']['Moid']
    return moid