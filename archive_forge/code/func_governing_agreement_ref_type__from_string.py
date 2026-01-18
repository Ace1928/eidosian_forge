import saml2
from saml2 import SamlBase
def governing_agreement_ref_type__from_string(xml_string):
    return saml2.create_class_from_xml_string(GoverningAgreementRefType_, xml_string)