import os
from typing import TYPE_CHECKING, Any, Dict, Optional, Sequence, Type, Union
from .. import _dtypes
from ..base_types.media import Media
def assign_type(self, wb_type: '_dtypes.Type') -> '_dtypes.Type':
    valid_ids = self.params['valid_ids'].assign_type(wb_type)
    if not isinstance(valid_ids, _dtypes.InvalidType):
        return self
    return _dtypes.InvalidType()