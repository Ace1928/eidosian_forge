import saml2
from saml2 import md
def discovery_response_from_string(xml_string):
    return saml2.create_class_from_xml_string(DiscoveryResponse, xml_string)