from __future__ import annotations
import abc
import tempfile
from pathlib import Path, PurePath
from lazyops.utils.pooler import ThreadPoolV2
from lazyops.utils.logs import default_logger, null_logger, Logger
from typing import Any, Union, Optional, Type, Iterable, Callable, Dict, List, Tuple, TypeVar, TYPE_CHECKING
def convert_sources_to_target(self, sources: Iterable[InputSourceType], target: str, source_filename: Optional[str]=None, target_output: Optional[Any]=None, **kwargs) -> Dict[str, OutputType]:
    """
        Convert the sources to the target

        sources: ['/path/to/file.pdf', '/path/to/file2.pdf']
        target: '.docx'
        """
    if not self.enabled:
        raise NotImplementedError(f'{self.name} is not enabled')
    return self._convert_sources_to_target(sources=sources, target=target, source_filename=source_filename, target_output=target_output, **kwargs)