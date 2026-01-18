import saml2
from saml2 import SamlBase
def physical_verification_from_string(xml_string):
    return saml2.create_class_from_xml_string(PhysicalVerification, xml_string)