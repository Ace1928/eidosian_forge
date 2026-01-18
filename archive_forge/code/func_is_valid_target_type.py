from __future__ import annotations
import abc
import tempfile
from pathlib import Path, PurePath
from lazyops.utils.pooler import ThreadPoolV2
from lazyops.utils.logs import default_logger, null_logger, Logger
from typing import Any, Union, Optional, Type, Iterable, Callable, Dict, List, Tuple, TypeVar, TYPE_CHECKING
def is_valid_target_type(self, data: Optional[InputContentType]=None, **kwargs) -> bool:
    """
        Validate whether the data is a valid target type for conversion
        """
    if data is None:
        return False
    if not isinstance(data, str) and hasattr(data, 'suffix'):
        return data.suffix in self.target_exts
    if isinstance(data, str):
        if len(data) < 5:
            return data in self.target_exts if '.' in data else data in self.targets
        return self.detect_content_type(data, mime=True) in self.target_mime_types
    return self.is_valid_target_path(data, **kwargs)