import saml2
from saml2 import SamlBase
def fault_faultstring_from_string(xml_string):
    return saml2.create_class_from_xml_string(Fault_faultstring, xml_string)