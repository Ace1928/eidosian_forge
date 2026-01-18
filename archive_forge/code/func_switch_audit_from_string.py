import saml2
from saml2 import SamlBase
def switch_audit_from_string(xml_string):
    return saml2.create_class_from_xml_string(SwitchAudit, xml_string)