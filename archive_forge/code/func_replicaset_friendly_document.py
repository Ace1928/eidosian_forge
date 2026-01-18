from __future__ import absolute_import, division, print_function
import time
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
from ansible_collections.community.mongodb.plugins.module_utils.mongodb_common import (
def replicaset_friendly_document(members_document):
    """
    Returns a version of the members document with
    only the info this module requires: name & stateStr
    """
    friendly_document = {}
    for member in members_document:
        friendly_document[member['name']] = member['stateStr']
    return friendly_document