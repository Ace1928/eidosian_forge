import saml2
from saml2 import SamlBase
def attributed_unsigned_long_type__from_string(xml_string):
    return saml2.create_class_from_xml_string(AttributedUnsignedLongType_, xml_string)