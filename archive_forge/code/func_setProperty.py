from xml.sax._exceptions import *
from xml.sax.handler import feature_validation, feature_namespaces
from xml.sax.handler import feature_namespace_prefixes
from xml.sax.handler import feature_external_ges, feature_external_pes
from xml.sax.handler import feature_string_interning
from xml.sax.handler import property_xml_string, property_interning_dict
import sys
from xml.sax import xmlreader, saxutils, handler
def setProperty(self, name, value):
    if name == handler.property_lexical_handler:
        self._lex_handler_prop = value
        if self._parsing:
            self._reset_lex_handler_prop()
    elif name == property_interning_dict:
        self._interning = value
    elif name == property_xml_string:
        raise SAXNotSupportedException("Property '%s' cannot be set" % name)
    else:
        raise SAXNotRecognizedException("Property '%s' not recognized" % name)