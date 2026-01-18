import typing
import functools
import pathlib
import importlib.util
from lazyops.utils.lazy import lazy_import
from typing import Optional, Dict, List, Union, Any, Callable, Type, TYPE_CHECKING
@functools.lru_cache()
def load_file_content(path: pathlib.Path, model: Optional[Type['BaseModel']]=None, loader: Optional[Callable]=None, binary_load: Optional[bool]=None, **kwargs) -> Any:
    """
    Load file content.

    args:
        path: path to the file
        model: model to parse the file with (default: None)
        loader: loader to use (default: None)
        binary_load: load the file as binary (default: None)
    """
    assert path.exists(), f'File {path} does not exist'
    binary_load = path.suffix in _binary_extensions if binary_load is None else binary_load
    if loader is None:
        loader = get_file_loader(path)
    data = path.read_bytes() if binary_load else path.read_text()
    if loader is None and model is None:
        return data
    if loader is not None:
        if path.suffix in {'jsonl', 'jsonlines'}:
            data = [loader(d) for d in data.splitlines()]
        else:
            data = loader(data)
    if model is not None:
        from lazyops.types.models import pyd_parse_obj
        data = pyd_parse_obj(model, data)
    return data