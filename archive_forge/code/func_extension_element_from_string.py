import logging
from typing import Any
from typing import Optional
from typing import Union
from xml.etree import ElementTree
import defusedxml.ElementTree
from saml2.validate import valid_instance
from saml2.version import version as __version__
def extension_element_from_string(xml_string):
    element_tree = defusedxml.ElementTree.fromstring(xml_string)
    return _extension_element_from_element_tree(element_tree)