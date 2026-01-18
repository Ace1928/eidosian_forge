import saml2
from saml2 import SamlBase
from saml2 import xmldsig as ds
def data_reference_from_string(xml_string):
    return saml2.create_class_from_xml_string(DataReference, xml_string)