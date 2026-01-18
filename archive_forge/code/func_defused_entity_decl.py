from __future__ import print_function, absolute_import
from xml.sax.expatreader import ExpatParser as _ExpatParser
from .common import DTDForbidden, EntitiesForbidden, ExternalReferenceForbidden
def defused_entity_decl(self, name, is_parameter_entity, value, base, sysid, pubid, notation_name):
    raise EntitiesForbidden(name, value, base, sysid, pubid, notation_name)