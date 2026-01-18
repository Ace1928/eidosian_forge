import saml2
from saml2 import md
def request_initiator_from_string(xml_string):
    return saml2.create_class_from_xml_string(RequestInitiator, xml_string)