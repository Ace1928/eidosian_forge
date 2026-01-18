import codecs
import collections
import decimal
import enum
import hashlib
import re
import uuid
from io import BytesIO, FileIO, IOBase
from pathlib import Path
from types import TracebackType
from typing import (
from ._cmap import build_char_map_from_dict
from ._doc_common import PdfDocCommon
from ._encryption import EncryptAlgorithm, Encryption
from ._page import PageObject
from ._page_labels import nums_clear_range, nums_insert, nums_next
from ._reader import PdfReader
from ._utils import (
from .constants import AnnotationDictionaryAttributes as AA
from .constants import CatalogAttributes as CA
from .constants import (
from .constants import CatalogDictionary as CD
from .constants import Core as CO
from .constants import (
from .constants import PageAttributes as PG
from .constants import PagesAttributes as PA
from .constants import TrailerKeys as TK
from .errors import PyPdfError
from .generic import (
from .pagerange import PageRange, PageRangeSpec
from .types import (
from .xmp import XmpInformation
def remove_images(self, to_delete: ImageType=ImageType.ALL) -> None:
    """
        Remove images from this output.

        Args:
            to_delete : The type of images to be deleted
                (default = all images types)
        """
    if isinstance(to_delete, bool):
        to_delete = ImageType.ALL
    i = (ObjectDeletionFlag.XOBJECT_IMAGES if to_delete & ImageType.XOBJECT_IMAGES else ObjectDeletionFlag.NONE) | (ObjectDeletionFlag.INLINE_IMAGES if to_delete & ImageType.INLINE_IMAGES else ObjectDeletionFlag.NONE) | (ObjectDeletionFlag.DRAWING_IMAGES if to_delete & ImageType.DRAWING_IMAGES else ObjectDeletionFlag.NONE)
    for page in self.pages:
        self.remove_objects_from_page(page, i)