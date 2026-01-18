import logging
from typing import Any
from typing import Optional
from typing import Union
from xml.etree import ElementTree
import defusedxml.ElementTree
from saml2.validate import valid_instance
from saml2.version import version as __version__
def fixup_element_prefixes(self, elem, uri_map, memo):

    def fixup(name):
        try:
            return memo[name]
        except KeyError:
            if name[0] != '{':
                return
            uri, tag = name[1:].split('}')
            if uri in uri_map:
                new_name = f'{uri_map[uri]}:{tag}'
                memo[name] = new_name
                return new_name
    name = fixup(elem.tag)
    if name:
        elem.tag = name
    for key, value in elem.items():
        name = fixup(key)
        if name:
            elem.set(name, value)
            del elem.attrib[key]