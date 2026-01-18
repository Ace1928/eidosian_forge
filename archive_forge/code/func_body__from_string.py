import saml2
from saml2 import SamlBase
def body__from_string(xml_string):
    return saml2.create_class_from_xml_string(Body_, xml_string)