import saml2
from saml2 import SamlBase
def security_from_string(xml_string):
    return saml2.create_class_from_xml_string(Security, xml_string)