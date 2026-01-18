from __future__ import (absolute_import, division, print_function)
import copy
import json
import os
import re
import traceback
from io import BytesIO
from ansible_collections.community.general.plugins.module_utils.version import LooseVersion
from ansible.module_utils.basic import AnsibleModule, json_dict_bytes_to_unicode, missing_required_lib
from ansible.module_utils.six import iteritems, string_types
from ansible.module_utils.common.text.converters import to_bytes, to_native
from ansible.module_utils.common._collections_compat import MutableMapping
def insert_target_children(tree, xpath, namespaces, children, insertbefore, insertafter):
    """
    Insert the given children before or after the given xpath. If insertbefore is True, it is inserted before the
    first xpath hit, with insertafter, it is inserted after the last xpath hit.
    """
    insert_target = tree.xpath(xpath, namespaces=namespaces)
    loc_index = 0 if insertbefore else -1
    index_in_parent = insert_target[loc_index].getparent().index(insert_target[loc_index])
    parent = insert_target[0].getparent()
    if insertafter:
        index_in_parent += 1
    for child in children:
        parent.insert(index_in_parent, child)
        index_in_parent += 1