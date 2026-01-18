import saml2
from saml2 import SamlBase
from saml2.ws import wsaddr as wsa
from saml2.ws import wssec as wsse
from saml2.ws import wsutil as wsu
def status_code_open_enum__from_string(xml_string):
    return saml2.create_class_from_xml_string(StatusCodeOpenEnum_, xml_string)