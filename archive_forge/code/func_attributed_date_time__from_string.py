import saml2
from saml2 import SamlBase
def attributed_date_time__from_string(xml_string):
    return saml2.create_class_from_xml_string(AttributedDateTime_, xml_string)