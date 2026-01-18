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
def get_named_dest_root(self) -> ArrayObject:
    named_dest = ArrayObject()
    if CA.NAMES in self.root_object and isinstance(self.root_object[CA.NAMES], DictionaryObject):
        names = cast(DictionaryObject, self.root_object[CA.NAMES])
        names_ref = names.indirect_reference
        if CA.DESTS in names and isinstance(names[CA.DESTS], DictionaryObject):
            dests = cast(DictionaryObject, names[CA.DESTS])
            dests_ref = dests.indirect_reference
            if CA.NAMES in dests:
                named_dest = cast(ArrayObject, dests[CA.NAMES])
            else:
                named_dest = ArrayObject()
                dests[NameObject(CA.NAMES)] = named_dest
        elif hasattr(self, '_add_object'):
            dests = DictionaryObject()
            dests_ref = self._add_object(dests)
            names[NameObject(CA.DESTS)] = dests_ref
            dests[NameObject(CA.NAMES)] = named_dest
    elif hasattr(self, '_add_object'):
        names = DictionaryObject()
        names_ref = self._add_object(names)
        self.root_object[NameObject(CA.NAMES)] = names_ref
        dests = DictionaryObject()
        dests_ref = self._add_object(dests)
        names[NameObject(CA.DESTS)] = dests_ref
        dests[NameObject(CA.NAMES)] = named_dest
    return named_dest