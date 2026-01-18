import saml2
from saml2 import SamlBase
def retrieval_method_from_string(xml_string):
    return saml2.create_class_from_xml_string(RetrievalMethod, xml_string)