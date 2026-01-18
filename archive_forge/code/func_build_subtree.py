from __future__ import absolute_import, division, print_function
import sys
from ansible.module_utils._text import to_bytes, to_text
from ansible.module_utils.connection import Connection, ConnectionError
def build_subtree(parent, path):
    element = parent
    for field in path.split('/'):
        sub_element = build_child_xml_node(element, field)
        element = sub_element
    return element