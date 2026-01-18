import typing
import functools
import pathlib
import importlib.util
from lazyops.utils.lazy import lazy_import
from typing import Optional, Dict, List, Union, Any, Callable, Type, TYPE_CHECKING
@functools.lru_cache()
def search_for_assets(path: pathlib.Path, allowed_extensions: typing.Optional[typing.List[str]]=None, recursive: typing.Optional[bool]=False, **kwargs) -> typing.List[pathlib.Path]:
    """
    Search for assets in a path.

    args:
        path: path to search
        allowed_extensions: list of allowed extensions (default: None)
        recursive: search recursively (default: False)
    
    Use it like this:

    >>> search_for_assets(pathlib.Path('lazyops/assets'), allowed_extensions = ['.yaml', '.yml'])
    """
    if allowed_extensions is None:
        allowed_extensions = _allowed_extensions
    result = []
    for file in path.iterdir():
        if file.is_dir() and recursive:
            result.extend(search_for_assets(file, allowed_extensions=allowed_extensions))
        elif file.suffix in allowed_extensions:
            result.append(file)
    return result