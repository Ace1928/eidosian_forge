import typing
import functools
import pathlib
import importlib.util
from lazyops.utils.lazy import lazy_import
from typing import Optional, Dict, List, Union, Any, Callable, Type, TYPE_CHECKING
def import_assets_wrapper(*path_parts, model: Optional[Type['BaseModel']]=None, load_file: Optional[bool]=False, **kwargs) -> Union[Dict[str, Any], List[Any]]:
    """
        Import assets from a module.

        args:
            path_parts: path parts to the assets directory (default: [])
            model: model to parse the assets with (default: None)
            load_file: load the file (default: False)
            **kwargs: additional arguments to pass to import_module_assets
        
        """
    return import_module_assets(module_name, *path_parts, assets_dir=assets_dir, model=model, load_file=load_file, allowed_extensions=allowed_extensions, recursive=recursive, as_dict=as_dict, **kwargs)