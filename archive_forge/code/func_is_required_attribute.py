import logging
from typing import Any
from typing import Optional
from typing import Union
from xml.etree import ElementTree
import defusedxml.ElementTree
from saml2.validate import valid_instance
from saml2.version import version as __version__
def is_required_attribute(cls, attr):
    """
    Check if the attribute is a required attribute for a specific SamlBase
    class.

    :param cls: The class
    :param attr: An attribute, note it must be the name of the attribute
        that appears in the XSD in which the class is defined.
    :return: True if required
    """
    return cls.c_attributes[attr][REQUIRED]