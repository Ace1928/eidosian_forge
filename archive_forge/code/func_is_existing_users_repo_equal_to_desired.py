from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule, missing_required_lib
from ansible_collections.infinidat.infinibox.plugins.module_utils.infinibox import (
@api_wrapper
def is_existing_users_repo_equal_to_desired(module):
    """ Compare two user user repositories. Return a bool. """
    newdata = create_post_data(module)
    olddata = get_users_repository(module, disable_fail=True)[0]
    if not olddata:
        return False
    if olddata['bind_username'] != newdata['bind_username']:
        return False
    if olddata['repository_type'] != newdata['repository_type']:
        return False
    if olddata['domain_name'] != newdata['domain_name']:
        return False
    if olddata['ldap_port'] != newdata['ldap_port']:
        return False
    if olddata['name'] != newdata['name']:
        return False
    if olddata['schema_definition'] != newdata['schema_definition']:
        return False
    if olddata['servers'] != newdata['servers']:
        return False
    if olddata['use_ldaps'] != newdata['use_ldaps']:
        return False
    return True