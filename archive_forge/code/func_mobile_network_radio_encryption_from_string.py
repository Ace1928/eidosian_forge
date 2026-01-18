import saml2
from saml2 import SamlBase
def mobile_network_radio_encryption_from_string(xml_string):
    return saml2.create_class_from_xml_string(MobileNetworkRadioEncryption, xml_string)