import saml2
from saml2 import SamlBase
from saml2.schema import wsdl
def t_header__from_string(xml_string):
    return saml2.create_class_from_xml_string(THeader_, xml_string)