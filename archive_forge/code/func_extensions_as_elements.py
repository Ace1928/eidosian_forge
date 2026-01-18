import logging
from typing import Any
from typing import Optional
from typing import Union
from xml.etree import ElementTree
import defusedxml.ElementTree
from saml2.validate import valid_instance
from saml2.version import version as __version__
def extensions_as_elements(self, tag, schema):
    """Return extensions that has the given tag and belongs to the
        given schema as native elements of that schema.

        :param tag: The tag of the element
        :param schema: Which schema the element should originate from
        :return: a list of native elements
        """
    result = []
    for ext in self.find_extensions(tag, schema.NAMESPACE):
        ets = schema.ELEMENT_FROM_STRING[tag]
        result.append(ets(ext.to_string()))
    return result