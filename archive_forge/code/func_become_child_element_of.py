import logging
from typing import Any
from typing import Optional
from typing import Union
from xml.etree import ElementTree
import defusedxml.ElementTree
from saml2.validate import valid_instance
from saml2.version import version as __version__
def become_child_element_of(self, node):
    """
        Note: Only for use with classes that have a c_tag and c_namespace class
        member. It is in SamlBase so that it can be inherited but it should
        not be called on instances of SamlBase.

        :param node: The node to which this instance should be a child
        """
    new_child = self._to_element_tree()
    node.append(new_child)