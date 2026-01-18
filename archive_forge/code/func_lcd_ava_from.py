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
def lcd_ava_from(self, attribute):
    """
        If nothing else works, this should

        :param attribute: an Attribute instance
        :return:
        """
    name = attribute.name.strip()
    values = [(value.text or '').strip() for value in attribute.attribute_value]
    return (name, values)