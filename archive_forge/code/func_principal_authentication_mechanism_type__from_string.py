import saml2
from saml2 import SamlBase
def principal_authentication_mechanism_type__from_string(xml_string):
    return saml2.create_class_from_xml_string(PrincipalAuthenticationMechanismType_, xml_string)