import saml2
from saml2 import SamlBase
def endpoint_reference_from_string(xml_string):
    return saml2.create_class_from_xml_string(EndpointReference, xml_string)