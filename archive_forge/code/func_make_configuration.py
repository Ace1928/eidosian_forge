from __future__ import absolute_import, division, print_function
import json
import logging
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.netapp_eseries.santricity.plugins.module_utils.netapp import request, eseries_host_argument_spec
from ansible.module_utils._text import to_native
def make_configuration(self):
    if not self.identifier:
        self.identifier = 'default'
    if not self.names:
        parts = urlparse.urlparse(self.server)
        netloc = parts.netloc
        if ':' in netloc:
            netloc = netloc.split(':')[0]
        self.names = [netloc]
    roles = list()
    for regex in self.role_mappings:
        for role in self.role_mappings[regex]:
            roles.append(dict(groupRegex=regex, ignoreCase=True, name=role))
    domain = dict(id=self.identifier, ldapUrl=self.server, bindLookupUser=dict(user=self.username, password=self.password), roleMapCollection=roles, groupAttributes=self.attributes, names=self.names, searchBase=self.search_base, userAttribute=self.user_attribute)
    return domain