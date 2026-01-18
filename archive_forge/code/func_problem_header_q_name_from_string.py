import saml2
from saml2 import SamlBase
def problem_header_q_name_from_string(xml_string):
    return saml2.create_class_from_xml_string(ProblemHeaderQName, xml_string)