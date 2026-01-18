import saml2
from saml2 import SamlBase
def hmac_output_length_from_string(xml_string):
    return saml2.create_class_from_xml_string(HMACOutputLength, xml_string)