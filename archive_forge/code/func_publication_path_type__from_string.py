import saml2
from saml2 import SamlBase
from saml2 import md
def publication_path_type__from_string(xml_string):
    return saml2.create_class_from_xml_string(PublicationPathType_, xml_string)