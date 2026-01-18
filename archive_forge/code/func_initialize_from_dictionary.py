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
@staticmethod
def initialize_from_dictionary(data: Dict[str, Any]) -> Union['EncodedStreamObject', 'DecodedStreamObject']:
    retval: Union[EncodedStreamObject, DecodedStreamObject]
    if SA.FILTER in data:
        retval = EncodedStreamObject()
    else:
        retval = DecodedStreamObject()
    retval._data = data['__streamdata__']
    del data['__streamdata__']
    del data[SA.LENGTH]
    retval.update(data)
    return retval