from __future__ import absolute_import, division, print_function
import json
import re
from difflib import Differ
from ansible.module_utils._text import to_bytes, to_text
from ansible.module_utils.connection import Connection, ConnectionError
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.netconf import (
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.utils import to_list
def build_xml_subtree(container_ele, xmap, param=None, opcode=None):
    sub_root = container_ele
    meta_subtree = list()
    for key, meta in xmap.items():
        candidates = meta.get('xpath', '').split('/')
        if container_ele.tag == candidates[-2]:
            parent = container_ele
        elif sub_root.tag == candidates[-2]:
            parent = sub_root
        else:
            parent = sub_root.find('.//' + meta.get('xpath', '').split(sub_root.tag + '/', 1)[1].rsplit('/', 1)[0])
        if opcode in ('delete', 'merge') and meta.get('operation', 'unknown') == 'edit' or meta.get('operation', None) is None:
            if meta.get('tag', False) is True:
                if parent.tag == container_ele.tag:
                    if meta.get('ns', False) is True:
                        child = etree.Element(candidates[-1], nsmap=NS_DICT[key.upper() + '_NSMAP'])
                    else:
                        child = etree.Element(candidates[-1])
                    meta_subtree.append(child)
                    sub_root = child
                elif meta.get('ns', False) is True:
                    child = etree.SubElement(parent, candidates[-1], nsmap=NS_DICT[key.upper() + '_NSMAP'])
                else:
                    child = etree.SubElement(parent, candidates[-1])
                if meta.get('attrib', None) is not None and opcode in ('delete', 'merge'):
                    child.set(BASE_1_0 + meta.get('attrib'), opcode)
                continue
            text = None
            param_key = key.split(':')
            if param_key[0] == 'a':
                if param is not None and param.get(param_key[1], None) is not None:
                    text = param.get(param_key[1])
            elif param_key[0] == 'm':
                if meta.get('value', None) is not None:
                    text = meta.get('value')
            if text:
                if meta.get('ns', False) is True:
                    child = etree.SubElement(parent, candidates[-1], nsmap=NS_DICT[key.upper() + '_NSMAP'])
                else:
                    child = etree.SubElement(parent, candidates[-1])
                child.text = text
                if meta.get('attrib', None) is not None and opcode in ('delete', 'merge'):
                    child.set(BASE_1_0 + meta.get('attrib'), opcode)
    if len(meta_subtree) > 1:
        for item in meta_subtree:
            container_ele.append(item)
    if sub_root == container_ele:
        return None
    else:
        return sub_root