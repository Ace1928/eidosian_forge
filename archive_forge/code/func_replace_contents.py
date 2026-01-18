import math
import re
import sys
from decimal import Decimal
from pathlib import Path
from typing import (
from ._cmap import build_char_map, unknown_char_map
from ._protocols import PdfCommonDocProtocol
from ._text_extraction import (
from ._utils import (
from .constants import AnnotationDictionaryAttributes as ADA
from .constants import ImageAttributes as IA
from .constants import PageAttributes as PG
from .constants import Resources as RES
from .errors import PageSizeNotDefinedError, PdfReadError
from .filters import _xobj_to_image
from .generic import (
def replace_contents(self, content: Union[None, ContentStream, EncodedStreamObject, ArrayObject]) -> None:
    """
        Replace the page contents with the new content and nullify old objects
        Args:
            content : new content. if None delete the content field.
        """
    if not hasattr(self, 'indirect_reference') or self.indirect_reference is None:
        self[NameObject(PG.CONTENTS)] = content
        return
    if isinstance(self.get(PG.CONTENTS, None), ArrayObject):
        for o in self[PG.CONTENTS]:
            try:
                self._objects[o.indirect_reference.idnum - 1] = NullObject()
            except AttributeError:
                pass
    if isinstance(content, ArrayObject):
        for i in range(len(content)):
            content[i] = self.indirect_reference.pdf._add_object(content[i])
    if content is None:
        if PG.CONTENTS not in self:
            return
        else:
            assert self.indirect_reference is not None
            assert self[PG.CONTENTS].indirect_reference is not None
            self.indirect_reference.pdf._objects[self[PG.CONTENTS].indirect_reference.idnum - 1] = NullObject()
            del self[PG.CONTENTS]
    elif not hasattr(self.get(PG.CONTENTS, None), 'indirect_reference'):
        try:
            self[NameObject(PG.CONTENTS)] = self.indirect_reference.pdf._add_object(content)
        except AttributeError:
            self[NameObject(PG.CONTENTS)] = content
    else:
        content.indirect_reference = self[PG.CONTENTS].indirect_reference
        try:
            self.indirect_reference.pdf._objects[content.indirect_reference.idnum - 1] = content
        except AttributeError:
            self[NameObject(PG.CONTENTS)] = content