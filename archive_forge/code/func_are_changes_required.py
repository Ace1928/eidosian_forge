from __future__ import absolute_import, division, print_function
from ansible_collections.netapp_eseries.santricity.plugins.module_utils.santricity import NetAppESeriesModule
from ansible.module_utils._text import to_native
def are_changes_required(self):
    """Determine whether any changes are required and build request body."""
    change_required = False
    domains = self.get_domains()
    if self.state == 'disabled' and domains:
        self.existing_domain_ids = [domain['id'] for domain in domains]
        change_required = True
    elif self.state == 'present':
        for domain in domains:
            if self.id == domain['id']:
                self.domain = domain
                if self.state == 'absent':
                    change_required = True
                elif len(self.group_attributes) != len(domain['groupAttributes']) or any([a not in domain['groupAttributes'] for a in self.group_attributes]):
                    change_required = True
                elif self.user_attribute != domain['userAttribute']:
                    change_required = True
                elif self.search_base.lower() != domain['searchBase'].lower():
                    change_required = True
                elif self.server != domain['ldapUrl']:
                    change_required = True
                elif any((name not in domain['names'] for name in self.names)) or any((name not in self.names for name in domain['names'])):
                    change_required = True
                elif self.role_mappings:
                    if len(self.body['roleMapCollection']) != len(domain['roleMapCollection']):
                        change_required = True
                    else:
                        for role_map in self.body['roleMapCollection']:
                            for existing_role_map in domain['roleMapCollection']:
                                if role_map['groupRegex'] == existing_role_map['groupRegex'] and role_map['name'] == existing_role_map['name']:
                                    break
                            else:
                                change_required = True
                if not change_required and self.bind_user and self.bind_password:
                    if self.bind_user != domain['bindLookupUser']['user']:
                        change_required = True
                    elif self.bind_password:
                        temporary_domain = None
                        try:
                            if any((domain['id'] == self.TEMPORARY_DOMAIN for domain in domains)):
                                self.delete_domain(self.TEMPORARY_DOMAIN)
                            temporary_domain = self.add_domain(temporary=True, skip_test=True)
                            rc, tests = self.request(self.url_path_prefix + 'ldap/test', method='POST')
                            temporary_domain_test = {}
                            domain_test = {}
                            for test in tests:
                                if test['id'] == temporary_domain['id']:
                                    temporary_domain_test = test['result']
                                if self.id == test['id']:
                                    domain_test = test['result']
                            if temporary_domain_test['authenticationTestResult'] == 'ok' and domain_test['authenticationTestResult'] != 'ok':
                                change_required = True
                            elif temporary_domain_test['authenticationTestResult'] != 'ok':
                                self.module.fail_json(msg='Failed to authenticate bind credentials! Array Id [%s].' % self.ssid)
                        finally:
                            if temporary_domain:
                                self.delete_domain(self.TEMPORARY_DOMAIN)
                break
        else:
            change_required = True
    elif self.state == 'absent':
        for domain in domains:
            if self.id == domain['id']:
                change_required = True
    return change_required