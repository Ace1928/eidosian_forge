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
@property
def mapping_name(self) -> Optional[str]:
    """
        Read-only property accessing the mapping name of this field.

        This name is used by pypdf as a key in the dictionary returned by
        :meth:`get_fields()<pypdf.PdfReader.get_fields>`
        """
    return self.get(FieldDictionaryAttributes.TM)