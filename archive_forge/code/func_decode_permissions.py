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
def decode_permissions(self, permissions_code: int) -> Dict[str, bool]:
    """Take the permissions as an integer, return the allowed access."""
    deprecate_with_replacement(old_name='decode_permissions', new_name='user_access_permissions', removed_in='5.0.0')
    permissions_mapping = {'print': UserAccessPermissions.PRINT, 'modify': UserAccessPermissions.MODIFY, 'copy': UserAccessPermissions.EXTRACT, 'annotations': UserAccessPermissions.ADD_OR_MODIFY, 'forms': UserAccessPermissions.FILL_FORM_FIELDS, 'accessability': UserAccessPermissions.EXTRACT_TEXT_AND_GRAPHICS, 'assemble': UserAccessPermissions.ASSEMBLE_DOC, 'print_high_quality': UserAccessPermissions.PRINT_TO_REPRESENTATION}
    return {key: permissions_code & flag != 0 for key, flag in permissions_mapping.items()}