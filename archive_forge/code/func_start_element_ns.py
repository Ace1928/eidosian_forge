from xml.sax._exceptions import *
from xml.sax.handler import feature_validation, feature_namespaces
from xml.sax.handler import feature_namespace_prefixes
from xml.sax.handler import feature_external_ges, feature_external_pes
from xml.sax.handler import feature_string_interning
from xml.sax.handler import property_xml_string, property_interning_dict
import sys
from xml.sax import xmlreader, saxutils, handler
def start_element_ns(self, name, attrs):
    pair = name.split()
    if len(pair) == 1:
        pair = (None, name)
    elif len(pair) == 3:
        pair = (pair[0], pair[1])
    else:
        pair = tuple(pair)
    newattrs = {}
    qnames = {}
    for aname, value in attrs.items():
        parts = aname.split()
        length = len(parts)
        if length == 1:
            qname = aname
            apair = (None, aname)
        elif length == 3:
            qname = '%s:%s' % (parts[2], parts[1])
            apair = (parts[0], parts[1])
        else:
            qname = parts[1]
            apair = tuple(parts)
        newattrs[apair] = value
        qnames[apair] = qname
    self._cont_handler.startElementNS(pair, None, AttributesNSImpl(newattrs, qnames))