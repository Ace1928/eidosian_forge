from __future__ import absolute_import, division, print_function
import re
import socket
import sys
import traceback
from ansible.module_utils.basic import env_fallback
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.utils import to_list, ComplexList
from ansible.module_utils.connection import exec_command, ConnectionError
from ansible.module_utils.six import iteritems
from ansible.module_utils._text import to_native
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.netconf import NetconfConnection
def merge_nc_xml(xml1, xml2):
    """merge xml1 and xml2"""
    xml1_list = xml1.split('</data>')[0].split('\n')
    xml2_list = xml2.split('<data>')[1].split('\n')
    while True:
        xml1_ele1 = get_xml_line(xml1_list, -1)
        xml1_ele2 = get_xml_line(xml1_list, -2)
        xml2_ele1 = get_xml_line(xml2_list, 0)
        xml2_ele2 = get_xml_line(xml2_list, 1)
        if not xml1_ele1 or not xml1_ele2 or (not xml2_ele1) or (not xml2_ele2):
            return xml1
        if 'xmlns' in xml2_ele1:
            xml2_ele1 = xml2_ele1.lstrip().split(' ')[0] + '>'
        if 'xmlns' in xml2_ele2:
            xml2_ele2 = xml2_ele2.lstrip().split(' ')[0] + '>'
        if xml1_ele1.replace(' ', '').replace('/', '') == xml2_ele1.replace(' ', '').replace('/', ''):
            if xml1_ele2.replace(' ', '').replace('/', '') == xml2_ele2.replace(' ', '').replace('/', ''):
                xml1_list.pop()
                xml2_list.pop(0)
            else:
                break
        else:
            break
    return '\n'.join(xml1_list + xml2_list)