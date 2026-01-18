from __future__ import print_function, absolute_import
from xml.sax.expatreader import ExpatParser as _ExpatParser
from .common import DTDForbidden, EntitiesForbidden, ExternalReferenceForbidden
def defused_start_doctype_decl(self, name, sysid, pubid, has_internal_subset):
    raise DTDForbidden(name, sysid, pubid)