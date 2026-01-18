import saml2
from saml2 import SamlBase
def dsa_key_value_type__from_string(xml_string):
    return saml2.create_class_from_xml_string(DSAKeyValueType_, xml_string)