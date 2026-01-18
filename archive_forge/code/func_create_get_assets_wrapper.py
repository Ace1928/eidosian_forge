import typing
import functools
import pathlib
import importlib.util
from lazyops.utils.lazy import lazy_import
from typing import Optional, Dict, List, Union, Any, Callable, Type, TYPE_CHECKING
def create_get_assets_wrapper(module_name: str, assets_dir: Optional[str]='assets', allowed_extensions: Optional[List[str]]=None, recursive: Optional[bool]=False, as_dict: Optional[bool]=False) -> Callable[..., Union[pathlib.Path, Any, List[pathlib.Path], List[Any], Dict[str, pathlib.Path], Dict[str, Any]]]:
    """
    Create a wrapper around `get_module_assets`.

    args:
        module_name: name of the module to import from (e.g. 'lazyops')
        assets_dir: name of the assets directory (default: 'assets')
        allowed_extensions: list of allowed extensions (default: None)
        as_dict: return a dict instead of a list (default: False)

    """

    def get_assets_wrapper(*path_parts, load_file: Optional[bool]=False, loader: Optional[Callable]=None, **kwargs) -> Union[pathlib.Path, Any, List[pathlib.Path], List[Any], Dict[str, pathlib.Path], Dict[str, Any]]:
        """
        Import assets from a module.

        args:
            path_parts: path parts to the assets directory (default: [])
            load_file: load the file (default: False)
            loader: loader function to use (default: None)
            **kwargs: additional arguments to pass to `get_module_assets`
        
        """
        assets = get_module_assets(module_name, *path_parts, assets_dir=assets_dir, allowed_extensions=allowed_extensions, recursive=recursive, **kwargs)
        if as_dict:
            if not isinstance(assets, list):
                assets = [assets]
            return {asset.name: load_file_content(asset, loader=loader, **kwargs) if load_file else asset for asset in assets}
        if isinstance(assets, list):
            return [load_file_content(asset, loader=loader, **kwargs) if load_file else asset for asset in assets]
        return load_file_content(assets, loader=loader, **kwargs) if load_file else assets
    return get_assets_wrapper