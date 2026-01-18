import saml2
from saml2 import SamlBase
from saml2 import xmldsig as ds
def encrypted_data_from_string(xml_string):
    return saml2.create_class_from_xml_string(EncryptedData, xml_string)