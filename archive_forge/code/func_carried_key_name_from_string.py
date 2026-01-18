import saml2
from saml2 import SamlBase
from saml2 import xmldsig as ds
def carried_key_name_from_string(xml_string):
    return saml2.create_class_from_xml_string(CarriedKeyName, xml_string)