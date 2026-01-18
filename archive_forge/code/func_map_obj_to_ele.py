from __future__ import absolute_import, division, print_function
import collections
import json
from contextlib import contextmanager
from copy import deepcopy
from ansible.module_utils._text import to_text
from ansible.module_utils.basic import env_fallback
from ansible.module_utils.connection import Connection, ConnectionError
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.netconf import (
def map_obj_to_ele(module, want, top, value_map=None, param=None):
    if not HAS_LXML:
        module.fail_json(msg='lxml is not installed.')
    if not param:
        param = module.params
    root = Element('root')
    top_ele = top.split('/')
    ele = SubElement(root, top_ele[0])
    if len(top_ele) > 1:
        for item in top_ele[1:-1]:
            ele = SubElement(ele, item)
    container = ele
    state = param.get('state')
    active = param.get('active')
    if active:
        oper = 'active'
    else:
        oper = 'inactive'
    if container.tag != top_ele[-1]:
        node = SubElement(container, top_ele[-1])
    else:
        node = container
    for fxpath, attributes in want.items():
        for attr in attributes:
            tag_only = attr.get('tag_only', False)
            leaf_only = attr.get('leaf_only', False)
            value_req = attr.get('value_req', False)
            is_key = attr.get('is_key', False)
            parent_attrib = attr.get('parent_attrib', True)
            value = attr.get('value')
            field_top = attr.get('top')
            if state == 'absent' and (not (is_key or leaf_only)):
                continue
            if value_map and fxpath in value_map:
                value = value_map[fxpath].get(value)
            if value is not None or tag_only or leaf_only:
                ele = node
                if field_top:
                    ele_list = root.xpath(top + '/' + field_top)
                    if not len(ele_list):
                        fields = field_top.split('/')
                        ele = node
                        for item in fields:
                            inner_ele = root.xpath(top + '/' + item)
                            if len(inner_ele):
                                ele = inner_ele[0]
                            else:
                                ele = SubElement(ele, item)
                    else:
                        ele = ele_list[0]
                if value is not None and (not isinstance(value, bool)):
                    value = to_text(value, errors='surrogate_then_replace')
                if fxpath:
                    tags = fxpath.split('/')
                    for item in tags:
                        ele = SubElement(ele, item)
                if tag_only:
                    if state == 'present':
                        if not value:
                            ele.set('delete', 'delete')
                elif leaf_only:
                    if state == 'present':
                        ele.set(oper, oper)
                        ele.text = value
                    else:
                        ele.set('delete', 'delete')
                        if value_req:
                            ele.text = value
                        if is_key:
                            par = ele.getparent()
                            par.set('delete', 'delete')
                else:
                    ele.text = value
                    par = ele.getparent()
                    if parent_attrib:
                        if state == 'present':
                            if not par.attrib.get('replace'):
                                par.set('replace', 'replace')
                            if not par.attrib.get(oper):
                                par.set(oper, oper)
                        else:
                            par.set('delete', 'delete')
    return root.getchildren()[0]