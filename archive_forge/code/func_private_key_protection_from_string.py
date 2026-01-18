import saml2
from saml2 import SamlBase
def private_key_protection_from_string(xml_string):
    return saml2.create_class_from_xml_string(PrivateKeyProtection, xml_string)