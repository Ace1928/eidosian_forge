import saml2
from saml2 import SamlBase
def attributed_q_name_type__from_string(xml_string):
    return saml2.create_class_from_xml_string(AttributedQNameType_, xml_string)