import logging
import re
from io import BytesIO
from typing import Dict, List, Mapping, Optional, Sequence, Tuple, Union, cast
from . import settings
from .cmapdb import CMap
from .cmapdb import CMapBase
from .cmapdb import CMapDB
from .pdfcolor import PDFColorSpace
from .pdfcolor import PREDEFINED_COLORSPACE
from .pdfdevice import PDFDevice
from .pdfdevice import PDFTextSeq
from .pdffont import PDFCIDFont
from .pdffont import PDFFont
from .pdffont import PDFFontError
from .pdffont import PDFTrueTypeFont
from .pdffont import PDFType1Font
from .pdffont import PDFType3Font
from .pdfpage import PDFPage
from .pdftypes import PDFException
from .pdftypes import PDFObjRef
from .pdftypes import PDFStream
from .pdftypes import dict_value
from .pdftypes import list_value
from .pdftypes import resolve1
from .pdftypes import stream_value
from .psparser import KWD
from .psparser import LIT
from .psparser import PSEOF
from .psparser import PSKeyword
from .psparser import PSLiteral, PSTypeError
from .psparser import PSStackParser
from .psparser import PSStackType
from .psparser import keyword_name
from .psparser import literal_name
from .utils import MATRIX_IDENTITY
from .utils import Matrix, Point, PathSegment, Rect
from .utils import choplist
from .utils import mult_matrix
def do_Do(self, xobjid_arg: PDFStackT) -> None:
    """Invoke named XObject"""
    xobjid = cast(str, literal_name(xobjid_arg))
    try:
        xobj = stream_value(self.xobjmap[xobjid])
    except KeyError:
        if settings.STRICT:
            raise PDFInterpreterError('Undefined xobject id: %r' % xobjid)
        return
    log.debug('Processing xobj: %r', xobj)
    subtype = xobj.get('Subtype')
    if subtype is LITERAL_FORM and 'BBox' in xobj:
        interpreter = self.dup()
        bbox = cast(Rect, list_value(xobj['BBox']))
        matrix = cast(Matrix, list_value(xobj.get('Matrix', MATRIX_IDENTITY)))
        xobjres = xobj.get('Resources')
        if xobjres:
            resources = dict_value(xobjres)
        else:
            resources = self.resources.copy()
        self.device.begin_figure(xobjid, bbox, matrix)
        interpreter.render_contents(resources, [xobj], ctm=mult_matrix(matrix, self.ctm))
        self.device.end_figure(xobjid)
    elif subtype is LITERAL_IMAGE and 'Width' in xobj and ('Height' in xobj):
        self.device.begin_figure(xobjid, (0, 0, 1, 1), MATRIX_IDENTITY)
        self.device.render_image(xobjid, xobj)
        self.device.end_figure(xobjid)
    else:
        pass
    return