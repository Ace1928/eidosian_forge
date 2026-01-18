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
def check_or_make_target(module, tree, xpath, namespaces):
    inner_xpath, changes = split_xpath_last(xpath)
    if inner_xpath == xpath or changes is None:
        module.fail_json(msg="Can't process Xpath %s in order to spawn nodes! tree is %s" % (xpath, etree.tostring(tree, pretty_print=True)))
        return False
    changed = False
    if not is_node(tree, inner_xpath, namespaces):
        changed = check_or_make_target(module, tree, inner_xpath, namespaces)
    if is_node(tree, inner_xpath, namespaces) and changes:
        for eoa, eoa_value in changes:
            if eoa and eoa[0] != '@' and (eoa[0] != '/'):
                new_kids = children_to_nodes(module, [nsnameToClark(eoa, namespaces)], 'yaml')
                if eoa_value:
                    for nk in new_kids:
                        nk.text = eoa_value
                for node in tree.xpath(inner_xpath, namespaces=namespaces):
                    node.extend(new_kids)
                    changed = True
            elif eoa and eoa[0] == '/':
                element = eoa[1:]
                new_kids = children_to_nodes(module, [nsnameToClark(element, namespaces)], 'yaml')
                for node in tree.xpath(inner_xpath, namespaces=namespaces):
                    node.extend(new_kids)
                    for nk in new_kids:
                        for subexpr in eoa_value:
                            check_or_make_target(module, nk, './' + subexpr, namespaces)
                    changed = True
            elif eoa == '':
                for node in tree.xpath(inner_xpath, namespaces=namespaces):
                    if node.text != eoa_value:
                        node.text = eoa_value
                        changed = True
            elif eoa and eoa[0] == '@':
                attribute = nsnameToClark(eoa[1:], namespaces)
                for element in tree.xpath(inner_xpath, namespaces=namespaces):
                    changing = attribute not in element.attrib or element.attrib[attribute] != eoa_value
                    if changing:
                        changed = changed or changing
                        if eoa_value is None:
                            value = ''
                        else:
                            value = eoa_value
                        element.attrib[attribute] = value
            else:
                module.fail_json(msg='unknown tree transformation=%s' % etree.tostring(tree, pretty_print=True))
    return changed