import saml2
from saml2 import SamlBase
def secret_key_protection_type__from_string(xml_string):
    return saml2.create_class_from_xml_string(SecretKeyProtectionType_, xml_string)