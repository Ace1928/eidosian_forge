from __future__ import print_function, absolute_import
import threading
import warnings
from lxml import etree as _etree
from .common import DTDForbidden, EntitiesForbidden, NotSupportedError
def check_docinfo(elementtree, forbid_dtd=False, forbid_entities=True):
    """Check docinfo of an element tree for DTD and entity declarations

    The check for entity declarations needs lxml 3 or newer. lxml 2.x does
    not support dtd.iterentities().
    """
    docinfo = elementtree.docinfo
    if docinfo.doctype:
        if forbid_dtd:
            raise DTDForbidden(docinfo.doctype, docinfo.system_url, docinfo.public_id)
        if forbid_entities and (not LXML3):
            raise NotSupportedError('Unable to check for entity declarations in lxml 2.x')
    if forbid_entities:
        for dtd in (docinfo.internalDTD, docinfo.externalDTD):
            if dtd is None:
                continue
            for entity in dtd.iterentities():
                raise EntitiesForbidden(entity.name, entity.content, None, None, None, None)