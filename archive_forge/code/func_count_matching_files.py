from pathlib import Path
from typing import Callable, Iterable, Iterator, Optional, Sequence, TypeVar, Union
from langchain_community.document_loaders.blob_loaders.schema import Blob, BlobLoader
def count_matching_files(self) -> int:
    """Count files that match the pattern without loading them."""
    num = 0
    for _ in self._yield_paths():
        num += 1
    return num