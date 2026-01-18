from __future__ import absolute_import, division, print_function
from collections import defaultdict
from ansible.module_utils.basic import to_text, AnsibleModule
def remove_acl(consul, configuration):
    """
    Removes an ACL.
    :param consul: the consul client
    :param configuration: the run configuration
    :return: the output of the removal
    """
    token = configuration.token
    changed = consul.acl.info(token) is not None
    if changed:
        consul.acl.destroy(token)
    return Output(changed=changed, token=token, operation=REMOVE_OPERATION)