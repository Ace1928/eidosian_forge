import saml2
from saml2 import SamlBase
from saml2 import md
def list_of_strings__from_string(xml_string):
    return saml2.create_class_from_xml_string(ListOfStrings_, xml_string)