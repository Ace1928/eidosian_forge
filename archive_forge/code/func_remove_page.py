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
def remove_page(self, page: Union[int, PageObject, IndirectObject], clean: bool=False) -> None:
    """
        Remove page from pages list.

        Args:
            page: int / PageObject / IndirectObject
                PageObject : page to be removed. If the page appears many times
                only the first one will be removed

                IndirectObject: Reference to page to be removed

                int: Page number to be removed

            clean: replace PageObject with NullObject to prevent destination,
                annotation to reference a detached page
        """
    if self.flattened_pages is None:
        self._flatten()
    assert self.flattened_pages is not None
    if isinstance(page, IndirectObject):
        p = page.get_object()
        if not isinstance(p, PageObject):
            logger_warning('IndirectObject is not referencing a page', __name__)
            return
        page = p
    if not isinstance(page, int):
        try:
            page = self.flattened_pages.index(page)
        except ValueError:
            logger_warning('Cannot find page in pages', __name__)
            return
    if not 0 <= page < len(self.flattened_pages):
        logger_warning('Page number is out of range', __name__)
        return
    ind = self.pages[page].indirect_reference
    del self.pages[page]
    if clean and ind is not None:
        self._replace_object(ind, NullObject())