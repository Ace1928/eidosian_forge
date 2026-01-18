import logging
from typing import Any
from typing import Optional
from typing import Union
from xml.etree import ElementTree
import defusedxml.ElementTree
from saml2.validate import valid_instance
from saml2.version import version as __version__
def find_children(self, tag=None, namespace=None):
    """Searches child nodes for objects with the desired tag/namespace.

        Returns a list of extension elements within this object whose tag
        and/or namespace match those passed in. To find all children in
        a particular namespace, specify the namespace but not the tag name.
        If you specify only the tag, the result list may contain extension
        elements in multiple namespaces.

        :param tag: str (optional) The desired tag
        :param namespace: str (optional) The desired namespace

        :return: A list of elements whose tag and/or namespace match the
            parameters values
        """
    results = []
    if tag and namespace:
        for element in self.children:
            if element.tag == tag and element.namespace == namespace:
                results.append(element)
    elif tag and (not namespace):
        for element in self.children:
            if element.tag == tag:
                results.append(element)
    elif namespace and (not tag):
        for element in self.children:
            if element.namespace == namespace:
                results.append(element)
    else:
        for element in self.children:
            results.append(element)
    return results