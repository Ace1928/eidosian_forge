import saml2
from saml2 import SamlBase
def authenticator_transport_protocol_type__from_string(xml_string):
    return saml2.create_class_from_xml_string(AuthenticatorTransportProtocolType_, xml_string)