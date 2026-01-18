import saml2
from saml2 import SamlBase
from saml2 import saml
from saml2 import xmldsig as ds
def authn_query_type__from_string(xml_string):
    return saml2.create_class_from_xml_string(AuthnQueryType_, xml_string)