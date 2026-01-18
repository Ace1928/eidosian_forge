from __future__ import annotations
import abc
import tempfile
from pathlib import Path, PurePath
from lazyops.utils.pooler import ThreadPoolV2
from lazyops.utils.logs import default_logger, null_logger, Logger
from typing import Any, Union, Optional, Type, Iterable, Callable, Dict, List, Tuple, TypeVar, TYPE_CHECKING
def is_valid_source_type(self, data: Optional[InputSourceType]=None, **kwargs) -> bool:
    """
        Validate whether the data is a valid source type for conversion

        data: .pdf
        """
    if data is None:
        return False
    if isinstance(data, Path) or hasattr(data, 'suffix'):
        return data.suffix == self.source_ext
    if isinstance(data, str):
        if len(data) < 5:
            return data == self.source_ext if '.' in data else data == self.source
        return self.detect_content_type(data, mime=True) == self.source_mime_type
    return self.is_valid_source_path(data, **kwargs)