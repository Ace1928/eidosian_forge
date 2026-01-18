import logging
from typing import Any
from typing import Optional
from typing import Union
from xml.etree import ElementTree
import defusedxml.ElementTree
from saml2.validate import valid_instance
from saml2.version import version as __version__
def harvest_element_tree(self, tree):
    for child in tree:
        self._convert_element_tree_to_member(child)
    for attribute, value in iter(tree.attrib.items()):
        self._convert_element_attribute_to_member(attribute, value)
    self.text = tree.text