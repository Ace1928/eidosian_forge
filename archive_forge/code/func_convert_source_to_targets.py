from __future__ import annotations
import abc
import tempfile
from pathlib import Path, PurePath
from lazyops.utils.pooler import ThreadPoolV2
from lazyops.utils.logs import default_logger, null_logger, Logger
from typing import Any, Union, Optional, Type, Iterable, Callable, Dict, List, Tuple, TypeVar, TYPE_CHECKING
def convert_source_to_targets(self, source: InputSourceType, targets: List[str], source_filename: Optional[str]=None, target_output: Optional[Any]=None, **kwargs) -> Dict[str, OutputType]:
    """
        Convert the source to the targets

        source: /path/to/file.pdf
        targets: ['.docx', '.txt']
        """
    if not self.enabled:
        raise NotImplementedError(f'{self.name} is not enabled')
    return self._convert_source_to_targets(source=source, targets=targets, source_filename=source_filename, target_output=target_output, **kwargs)