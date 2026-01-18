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
def child_to_element(module, child, in_type):
    if in_type == 'xml':
        infile = BytesIO(to_bytes(child, errors='surrogate_or_strict'))
        try:
            parser = etree.XMLParser()
            node = etree.parse(infile, parser)
            return node.getroot()
        except etree.XMLSyntaxError as e:
            module.fail_json(msg='Error while parsing child element: %s' % e)
    elif in_type == 'yaml':
        if isinstance(child, string_types):
            return etree.Element(child)
        elif isinstance(child, MutableMapping):
            if len(child) > 1:
                module.fail_json(msg='Can only create children from hashes with one key')
            key, value = next(iteritems(child))
            if isinstance(value, MutableMapping):
                children = value.pop('_', None)
                node = etree.Element(key, value)
                if children is not None:
                    if not isinstance(children, list):
                        module.fail_json(msg='Invalid children type: %s, must be list.' % type(children))
                    subnodes = children_to_nodes(module, children)
                    node.extend(subnodes)
            else:
                node = etree.Element(key)
                node.text = value
            return node
        else:
            module.fail_json(msg='Invalid child type: %s. Children must be either strings or hashes.' % type(child))
    else:
        module.fail_json(msg='Invalid child input type: %s. Type must be either xml or yaml.' % in_type)