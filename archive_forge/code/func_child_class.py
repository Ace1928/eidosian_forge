import logging
from typing import Any
from typing import Optional
from typing import Union
from xml.etree import ElementTree
import defusedxml.ElementTree
from saml2.validate import valid_instance
from saml2.version import version as __version__
def child_class(self, child):
    """Return the class a child element should be an instance of

        :param child: The name of the child element
        :return: The class
        """
    for prop, klassdef in self.c_children.values():
        if child == prop:
            if isinstance(klassdef, list):
                return klassdef[0]
            else:
                return klassdef
    return None