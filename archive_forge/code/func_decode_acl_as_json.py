from __future__ import absolute_import, division, print_function
from collections import defaultdict
from ansible.module_utils.basic import to_text, AnsibleModule
def decode_acl_as_json(acl_as_json):
    """
    Converts the given JSON representation of an ACL into the equivalent domain model.
    :param acl_as_json: the JSON representation of an ACL
    :return: the equivalent domain model to the given ACL
    """
    rules_as_hcl = acl_as_json[_RULES_JSON_PROPERTY]
    rules = decode_rules_as_hcl_string(acl_as_json[_RULES_JSON_PROPERTY]) if rules_as_hcl.strip() != '' else RuleCollection()
    return ACL(rules=rules, token_type=acl_as_json[_TOKEN_TYPE_JSON_PROPERTY], token=acl_as_json[_TOKEN_JSON_PROPERTY], name=acl_as_json[_NAME_JSON_PROPERTY])