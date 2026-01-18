import struct
import zlib
from abc import abstractmethod
from datetime import datetime
from typing import (
from ._encryption import Encryption
from ._page import PageObject, _VirtualList
from ._page_labels import index2label as page_index2page_label
from ._utils import (
from .constants import CatalogAttributes as CA
from .constants import CatalogDictionary as CD
from .constants import (
from .constants import Core as CO
from .constants import DocumentInformationAttributes as DI
from .constants import FieldDictionaryAttributes as FA
from .constants import PageAttributes as PG
from .constants import PagesAttributes as PA
from .errors import (
from .generic import (
from .types import OutlineType, PagemodeType
from .xmp import XmpInformation
def get_num_pages(self) -> int:
    """
        Calculate the number of pages in this PDF file.

        Returns:
            The number of pages of the parsed PDF file

        Raises:
            PdfReadError: if file is encrypted and restrictions prevent
                this action.
        """
    if self.is_encrypted:
        return self.root_object['/Pages']['/Count']
    else:
        if self.flattened_pages is None:
            self._flatten()
        assert self.flattened_pages is not None
        return len(self.flattened_pages)