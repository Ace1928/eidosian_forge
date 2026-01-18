import logging
import re
from xml.etree import ElementTree as ElementTree
import defusedxml.ElementTree
from saml2 import create_class_from_element_tree
from saml2.samlp import NAMESPACE as SAMLP_NAMESPACE
from saml2.schema import soapenv
def parse_soap_enveloped_saml_attribute_query(text):
    expected_tag = '{%s}AttributeQuery' % SAMLP_NAMESPACE
    return parse_soap_enveloped_saml_thingy(text, [expected_tag])