from Bio import motifs
from xml.dom import minidom, Node
import re
def get_property_value(self, node, key_name):
    """Extract the value of the motif's property named key_name from node."""
    for cur_property in node.getElementsByTagName('prop'):
        right_property = False
        cur_value = None
        for child in cur_property.childNodes:
            if child.nodeType != Node.ELEMENT_NODE:
                continue
            if child.tagName == 'key' and self.get_text([child]) == key_name:
                right_property = True
            if child.tagName == 'value':
                cur_value = self.get_text([child])
        if right_property:
            return cur_value
    return None