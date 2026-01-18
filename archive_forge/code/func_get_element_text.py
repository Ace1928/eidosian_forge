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
def get_element_text(module, tree, xpath, namespaces):
    if not is_node(tree, xpath, namespaces):
        module.fail_json(msg='Xpath %s does not reference a node!' % xpath)
    elements = []
    for element in tree.xpath(xpath, namespaces=namespaces):
        elements.append({element.tag: element.text})
    finish(module, tree, xpath, namespaces, changed=False, msg=len(elements), hitcount=len(elements), matches=elements)