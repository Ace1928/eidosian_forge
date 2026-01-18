import saml2
from saml2 import SamlBase
from saml2 import saml
from saml2 import xmldsig as ds
def manage_name_id_request_type__from_string(xml_string):
    return saml2.create_class_from_xml_string(ManageNameIDRequestType_, xml_string)