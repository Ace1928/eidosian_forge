import saml2
from saml2 import SamlBase
def signature_value_type__from_string(xml_string):
    return saml2.create_class_from_xml_string(SignatureValueType_, xml_string)