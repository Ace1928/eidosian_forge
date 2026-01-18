import os
import re
from io import BytesIO, UnsupportedOperation
from pathlib import Path
from typing import (
from ._doc_common import PdfDocCommon, convert_to_int
from ._encryption import Encryption, PasswordType
from ._page import PageObject
from ._utils import (
from .constants import TrailerKeys as TK
from .errors import (
from .generic import (
from .xmp import XmpInformation
def cache_indirect_object(self, generation: int, idnum: int, obj: Optional[PdfObject]) -> Optional[PdfObject]:
    if (generation, idnum) in self.resolved_objects:
        msg = f'Overwriting cache for {generation} {idnum}'
        if self.strict:
            raise PdfReadError(msg)
        logger_warning(msg, __name__)
    self.resolved_objects[generation, idnum] = obj
    if obj is not None:
        obj.indirect_reference = IndirectObject(idnum, generation, self)
    return obj