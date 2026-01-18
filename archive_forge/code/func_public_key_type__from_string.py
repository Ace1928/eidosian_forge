import saml2
from saml2 import SamlBase
def public_key_type__from_string(xml_string):
    return saml2.create_class_from_xml_string(PublicKeyType_, xml_string)