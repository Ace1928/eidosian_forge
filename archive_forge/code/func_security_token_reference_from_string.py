import saml2
from saml2 import SamlBase
def security_token_reference_from_string(xml_string):
    return saml2.create_class_from_xml_string(SecurityTokenReference, xml_string)