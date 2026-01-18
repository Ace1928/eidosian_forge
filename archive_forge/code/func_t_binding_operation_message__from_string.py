import saml2
from saml2 import SamlBase
def t_binding_operation_message__from_string(xml_string):
    return saml2.create_class_from_xml_string(TBindingOperationMessage_, xml_string)