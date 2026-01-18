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
def additional_actions(self) -> Optional[DictionaryObject]:
    """
        Read-only property accessing the additional actions dictionary.

        This dictionary defines the field's behavior in response to trigger
        events. See Section 8.5.2 of the PDF 1.7 reference.
        """
    return self.get(FieldDictionaryAttributes.AA)