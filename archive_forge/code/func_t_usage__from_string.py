import saml2
from saml2 import SamlBase
def t_usage__from_string(xml_string):
    return saml2.create_class_from_xml_string(TUsage_, xml_string)