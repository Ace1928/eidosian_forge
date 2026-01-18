from __future__ import absolute_import, division, print_function
from collections import defaultdict
from ansible.module_utils.basic import to_text, AnsibleModule
def decode_acls_as_json(acls_as_json):
    """
    Converts the given JSON representation of ACLs into a list of ACL domain models.
    :param acls_as_json: the JSON representation of a collection of ACLs
    :return: list of equivalent domain models for the given ACLs (order not guaranteed to be the same)
    """
    return [decode_acl_as_json(acl_as_json) for acl_as_json in acls_as_json]