import saml2
from saml2 import SamlBase
from saml2 import md
def geolocation_hint_from_string(xml_string):
    return saml2.create_class_from_xml_string(GeolocationHint, xml_string)