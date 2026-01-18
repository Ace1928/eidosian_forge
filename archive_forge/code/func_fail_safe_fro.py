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
def fail_safe_fro(self, statement):
    """In case there is not formats defined or if the name format is
        undefined

        :param statement: AttributeStatement instance
        :return: A dictionary with names and values
        """
    result = {}
    for attribute in statement.attribute:
        if attribute.name_format and attribute.name_format != NAME_FORMAT_UNSPECIFIED:
            continue
        try:
            name = attribute.friendly_name.strip()
        except AttributeError:
            name = attribute.name.strip()
        result[name] = []
        for value in attribute.attribute_value:
            if not value.text:
                result[name].append('')
            else:
                result[name].append(value.text.strip())
    return result