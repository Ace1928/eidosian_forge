import saml2
from saml2 import SamlBase
def signature_value_from_string(xml_string):
    return saml2.create_class_from_xml_string(SignatureValue, xml_string)