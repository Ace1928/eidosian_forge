import collections
from abc import ABC, abstractmethod
from pathlib import Path
from typing import IO, Any, Callable, Dict, Iterator, List, Optional, Sequence, Union
from langchain_core.documents import Document
from langchain_community.document_loaders.base import BaseLoader
def satisfies_min_unstructured_version(min_version: str) -> bool:
    """Check if the installed `Unstructured` version exceeds the minimum version
    for the feature in question."""
    from unstructured.__version__ import __version__ as __unstructured_version__
    min_version_tuple = tuple([int(x) for x in min_version.split('.')])
    _unstructured_version = __unstructured_version__.split('-')[0]
    unstructured_version_tuple = tuple([int(x) for x in _unstructured_version.split('.')])
    return unstructured_version_tuple >= min_version_tuple