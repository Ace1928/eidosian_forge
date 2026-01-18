import saml2
from saml2 import SamlBase
def request_from_string(xml_string):
    return saml2.create_class_from_xml_string(Request, xml_string)