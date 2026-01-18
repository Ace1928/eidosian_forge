from __future__ import absolute_import, division, print_function
import time
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
from ansible_collections.community.mongodb.plugins.module_utils.mongodb_common import (
def replicaset_members(replicaset_document):
    """
    Returns the members section of the MongoDB replicaset document
    """
    return replicaset_document['members']