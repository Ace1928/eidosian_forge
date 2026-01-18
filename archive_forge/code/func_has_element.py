from __future__ import (absolute_import, division, print_function)
import xml.etree.ElementTree as ET
import re
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
from ansible_collections.community.network.plugins.module_utils.network.cloudengine.ce import get_nc_config, set_nc_config
def has_element(parent, xpath):
    """get or create a element by xpath"""
    ele = parent.find('./' + xpath)
    if ele is not None:
        return ele
    ele = parent
    lpath = xpath.split('/')
    for p in lpath:
        e = parent.find('.//' + p)
        if e is None:
            e = ET.SubElement(ele, p)
        ele = e
    return ele