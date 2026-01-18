import saml2
from saml2 import SamlBase
from saml2 import xmldsig as ds
def originator_key_info_from_string(xml_string):
    return saml2.create_class_from_xml_string(OriginatorKeyInfo, xml_string)