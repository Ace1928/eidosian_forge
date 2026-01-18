import saml2
from saml2 import SamlBase
from saml2.schema import wsdl
def binding_from_string(xml_string):
    return saml2.create_class_from_xml_string(Binding, xml_string)