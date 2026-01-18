import saml2
from saml2 import SamlBase
def encoded_string__from_string(xml_string):
    return saml2.create_class_from_xml_string(EncodedString_, xml_string)