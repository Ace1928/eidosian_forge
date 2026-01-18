import typing
import functools
import pathlib
import importlib.util
from lazyops.utils.lazy import lazy_import
from typing import Optional, Dict, List, Union, Any, Callable, Type, TYPE_CHECKING
@functools.lru_cache()
def get_module_assets(module_name: str, *path_parts, assets_dir: Optional[str]='assets', allowed_extensions: Optional[List[str]]=None, recursive: Optional[bool]=False, **kwargs) -> Union[pathlib.Path, List[pathlib.Path]]:
    """
    Get the path to an assets from a module.

    args:
        module_name: name of the module to import from (e.g. 'configz')
        path_parts: path parts to the assets directory (default: [])
        assets_dir: name of the assets directory (default: 'assets')
        allowed_extensions: list of allowed extensions (default: None)
    
    Use it like this:

    >>> get_module_assets_path('lazyops', 'authz', 'file.json', assets_dir = 'assets')
    >>> get_module_assets_path('lazyops', 'authz') 
    """
    module_assets_path = get_module_assets_path(module_name, assets_dir=assets_dir)
    module_assets = module_assets_path.joinpath(*path_parts)
    if module_assets.is_file():
        return module_assets
    return search_for_assets(module_assets, allowed_extensions=allowed_extensions, recursive=recursive)