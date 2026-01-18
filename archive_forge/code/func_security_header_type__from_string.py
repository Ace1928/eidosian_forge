import saml2
from saml2 import SamlBase
def security_header_type__from_string(xml_string):
    return saml2.create_class_from_xml_string(SecurityHeaderType_, xml_string)