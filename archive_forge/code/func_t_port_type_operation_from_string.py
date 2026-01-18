import saml2
from saml2 import SamlBase
def t_port_type_operation_from_string(xml_string):
    return saml2.create_class_from_xml_string(TPortType_operation, xml_string)