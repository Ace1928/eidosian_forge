from __future__ import absolute_import, division, print_function
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
import ansible_collections.netapp.ontap.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.ontap.plugins.module_utils.netapp_module import NetAppModule
from ansible_collections.netapp.ontap.plugins.module_utils import rest_generic, rest_vserver
def match_acl_with_acls(self, acl, acls):
    """ return acl if user and access and apply_to are matched, otherwiese None """
    matches = []
    for an_acl in acls:
        inherited = an_acl['inherited'] if 'inherited' in an_acl else False and (acl['inherited'] if 'inherited' in acl else False)
        if acl['user'] == an_acl['user'] and acl['access'] == an_acl['access'] and (acl.get('access_control', 'file_directory') == an_acl.get('access_control', 'file_directory')) and (acl['apply_to'] == an_acl['apply_to']) and (not inherited):
            matches.append(an_acl)
    if len(matches) > 1:
        self.module.fail_json(msg='Error matching ACLs, found more than one match.  Found %s' % matches)
    return matches[0] if matches else None