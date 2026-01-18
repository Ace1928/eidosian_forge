from __future__ import annotations
import abc
import tempfile
from pathlib import Path, PurePath
from lazyops.utils.pooler import ThreadPoolV2
from lazyops.utils.logs import default_logger, null_logger, Logger
from typing import Any, Union, Optional, Type, Iterable, Callable, Dict, List, Tuple, TypeVar, TYPE_CHECKING
def is_valid_target_content_type(self, content_type: Optional[str]=None, **kwargs) -> bool:
    """
        Validate whether the content type is a valid target content type for conversion

        content-type: application/pdf
        """
    if '.' in content_type:
        content_type = content_type.split('.')[-1]
    return any((t in content_type.lower() for t in self.targets)) if content_type else False