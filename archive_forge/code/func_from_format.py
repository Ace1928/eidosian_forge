from importlib import import_module
import logging
import os
import sys
from saml2 import NAMESPACE
from saml2 import ExtensionElement
from saml2 import SAMLError
from saml2 import extension_elements_to_elements
from saml2 import saml
from saml2.s_utils import do_ava
from saml2.s_utils import factory
from saml2.saml import NAME_FORMAT_UNSPECIFIED
from saml2.saml import NAMEID_FORMAT_PERSISTENT
def from_format(self, attr):
    """Find out the local name of an attribute

        :param attr: An saml.Attribute instance
        :return: The local attribute name or "" if no mapping could be made
        """
    if attr.name_format:
        if self.name_format == attr.name_format:
            try:
                return self._fro[attr.name.lower()]
            except KeyError:
                pass
    else:
        try:
            return self._fro[attr.name.lower()]
        except KeyError:
            pass
    return ''