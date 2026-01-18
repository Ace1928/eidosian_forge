import saml2
from saml2 import SamlBase
def deactivation_call_center_from_string(xml_string):
    return saml2.create_class_from_xml_string(DeactivationCallCenter, xml_string)