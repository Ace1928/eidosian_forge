import saml2
from saml2 import SamlBase
def pgp_key_id_from_string(xml_string):
    return saml2.create_class_from_xml_string(PGPKeyID, xml_string)