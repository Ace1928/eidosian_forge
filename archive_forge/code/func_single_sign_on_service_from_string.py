import saml2
from saml2 import SamlBase
from saml2 import saml
from saml2 import xmldsig as ds
from saml2 import xmlenc as xenc
def single_sign_on_service_from_string(xml_string):
    return saml2.create_class_from_xml_string(SingleSignOnService, xml_string)