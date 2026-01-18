import saml2
from saml2 import SamlBase
def key_sharing_from_string(xml_string):
    return saml2.create_class_from_xml_string(KeySharing, xml_string)