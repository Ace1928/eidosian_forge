import logging
from typing import Any
from typing import Optional
from typing import Union
from xml.etree import ElementTree
import defusedxml.ElementTree
from saml2.validate import valid_instance
from saml2.version import version as __version__
def set_prefixes(self, elem, prefix_map):
    if not ElementTree.iselement(elem):
        elem = elem.getroot()
    uri_map = {}
    for prefix, uri in prefix_map.items():
        uri_map[uri] = prefix
        elem.set(f'xmlns:{prefix}', uri)
    memo = {}
    for element in elem.iter():
        self.fixup_element_prefixes(element, uri_map, memo)