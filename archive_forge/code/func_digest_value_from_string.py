import saml2
from saml2 import SamlBase
def digest_value_from_string(xml_string):
    return saml2.create_class_from_xml_string(DigestValue, xml_string)