import saml2
from saml2 import SamlBase
def definitions_from_string(xml_string):
    return saml2.create_class_from_xml_string(Definitions, xml_string)