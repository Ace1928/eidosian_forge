import saml2
from saml2 import SamlBase
def restricted_password_type__from_string(xml_string):
    return saml2.create_class_from_xml_string(RestrictedPasswordType_, xml_string)