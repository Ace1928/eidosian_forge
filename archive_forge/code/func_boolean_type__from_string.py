import saml2
from saml2 import SamlBase
def boolean_type__from_string(xml_string):
    return saml2.create_class_from_xml_string(BooleanType_, xml_string)