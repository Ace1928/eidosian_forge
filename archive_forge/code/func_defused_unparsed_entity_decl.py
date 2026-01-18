from __future__ import print_function, absolute_import
from xml.sax.expatreader import ExpatParser as _ExpatParser
from .common import DTDForbidden, EntitiesForbidden, ExternalReferenceForbidden
def defused_unparsed_entity_decl(self, name, base, sysid, pubid, notation_name):
    raise EntitiesForbidden(name, None, base, sysid, pubid, notation_name)