import saml2
from saml2 import SamlBase
def t_service_port_from_string(xml_string):
    return saml2.create_class_from_xml_string(TService_port, xml_string)