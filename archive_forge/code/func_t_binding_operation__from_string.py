import saml2
from saml2 import SamlBase
def t_binding_operation__from_string(xml_string):
    return saml2.create_class_from_xml_string(TBindingOperation_, xml_string)