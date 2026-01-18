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
def init_resources(self, resources: Dict[object, object]) -> None:
    """Prepare the fonts and XObjects listed in the Resource attribute."""
    self.resources = resources
    self.fontmap: Dict[object, PDFFont] = {}
    self.xobjmap = {}
    self.csmap: Dict[str, PDFColorSpace] = PREDEFINED_COLORSPACE.copy()
    if not resources:
        return

    def get_colorspace(spec: object) -> Optional[PDFColorSpace]:
        if isinstance(spec, list):
            name = literal_name(spec[0])
        else:
            name = literal_name(spec)
        if name == 'ICCBased' and isinstance(spec, list) and (2 <= len(spec)):
            return PDFColorSpace(name, stream_value(spec[1])['N'])
        elif name == 'DeviceN' and isinstance(spec, list) and (2 <= len(spec)):
            return PDFColorSpace(name, len(list_value(spec[1])))
        else:
            return PREDEFINED_COLORSPACE.get(name)
    for k, v in dict_value(resources).items():
        log.debug('Resource: %r: %r', k, v)
        if k == 'Font':
            for fontid, spec in dict_value(v).items():
                objid = None
                if isinstance(spec, PDFObjRef):
                    objid = spec.objid
                spec = dict_value(spec)
                self.fontmap[fontid] = self.rsrcmgr.get_font(objid, spec)
        elif k == 'ColorSpace':
            for csid, spec in dict_value(v).items():
                colorspace = get_colorspace(resolve1(spec))
                if colorspace is not None:
                    self.csmap[csid] = colorspace
        elif k == 'ProcSet':
            self.rsrcmgr.get_procset(list_value(v))
        elif k == 'XObject':
            for xobjid, xobjstrm in dict_value(v).items():
                self.xobjmap[xobjid] = xobjstrm
    return