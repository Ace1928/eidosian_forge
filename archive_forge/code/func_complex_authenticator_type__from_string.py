import saml2
from saml2 import SamlBase
def complex_authenticator_type__from_string(xml_string):
    return saml2.create_class_from_xml_string(ComplexAuthenticatorType_, xml_string)