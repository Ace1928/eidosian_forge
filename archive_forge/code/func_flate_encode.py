import logging
import re
import sys
from io import BytesIO
from typing import (
from .._protocols import PdfReaderProtocol, PdfWriterProtocol, XmpInformationProtocol
from .._utils import (
from ..constants import (
from ..constants import FilterTypes as FT
from ..constants import StreamAttributes as SA
from ..constants import TypArguments as TA
from ..constants import TypFitArguments as TF
from ..errors import STREAM_TRUNCATED_PREMATURELY, PdfReadError, PdfStreamError
from ._base import (
from ._fit import Fit
from ._utils import read_hex_string_from_stream, read_string_from_stream
def flate_encode(self, level: int=-1) -> 'EncodedStreamObject':
    from ..filters import FlateDecode
    if SA.FILTER in self:
        f = self[SA.FILTER]
        if isinstance(f, ArrayObject):
            f = ArrayObject([NameObject(FT.FLATE_DECODE), *f])
            try:
                params = ArrayObject([NullObject(), *self.get(SA.DECODE_PARMS, ArrayObject())])
            except TypeError:
                params = ArrayObject([NullObject(), self.get(SA.DECODE_PARMS, ArrayObject())])
        else:
            f = ArrayObject([NameObject(FT.FLATE_DECODE), f])
            params = ArrayObject([NullObject(), self.get(SA.DECODE_PARMS, NullObject())])
    else:
        f = NameObject(FT.FLATE_DECODE)
        params = None
    retval = EncodedStreamObject()
    retval.update(self)
    retval[NameObject(SA.FILTER)] = f
    if params is not None:
        retval[NameObject(SA.DECODE_PARMS)] = params
    retval._data = FlateDecode.encode(b_(self._data), level)
    return retval