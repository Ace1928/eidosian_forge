import saml2
from saml2 import SamlBase
def sp_type_type__from_string(xml_string):
    return saml2.create_class_from_xml_string(SPTypeType_, xml_string)