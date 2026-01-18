import saml2
from saml2 import SamlBase
from saml2 import saml
from saml2 import xmldsig as ds
def status_detail_from_string(xml_string):
    return saml2.create_class_from_xml_string(StatusDetail, xml_string)