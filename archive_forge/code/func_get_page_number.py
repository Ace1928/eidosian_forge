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
def get_page_number(self, page: PageObject) -> Optional[int]:
    """
        Retrieve page number of a given PageObject.

        Args:
            page: The page to get page number. Should be
                an instance of :class:`PageObject<pypdf._page.PageObject>`

        Returns:
            The page number or None if page is not found
        """
    return self._get_page_number_by_indirect(page.indirect_reference)