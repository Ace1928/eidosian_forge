import saml2
from saml2 import SamlBase
def endpoint_reference_type__address_from_string(xml_string):
    return saml2.create_class_from_xml_string(EndpointReferenceType_Address, xml_string)