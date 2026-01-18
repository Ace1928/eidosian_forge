import saml2
from saml2 import SamlBase
def ip_address_from_string(xml_string):
    return saml2.create_class_from_xml_string(IPAddress, xml_string)