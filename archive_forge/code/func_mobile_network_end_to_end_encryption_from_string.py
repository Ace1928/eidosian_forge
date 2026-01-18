import saml2
from saml2 import SamlBase
def mobile_network_end_to_end_encryption_from_string(xml_string):
    return saml2.create_class_from_xml_string(MobileNetworkEndToEndEncryption, xml_string)