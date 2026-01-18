import saml2
from saml2 import SamlBase
def header__from_string(xml_string):
    return saml2.create_class_from_xml_string(Header_, xml_string)