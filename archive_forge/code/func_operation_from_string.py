import saml2
from saml2 import SamlBase
from saml2.schema import wsdl
def operation_from_string(xml_string):
    return saml2.create_class_from_xml_string(Operation, xml_string)