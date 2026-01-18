import typing
import functools
import pathlib
import importlib.util
from lazyops.utils.lazy import lazy_import
from typing import Optional, Dict, List, Union, Any, Callable, Type, TYPE_CHECKING
@functools.lru_cache()
def import_module_assets(module_name: str, *path_parts, model: Optional[Type['BaseModel']]=None, load_file: Optional[bool]=False, loader: Optional[Callable]=None, assets_dir: Optional[str]='assets', allowed_extensions: Optional[List[str]]=None, recursive: Optional[bool]=False, as_dict: Optional[bool]=False, **kwargs) -> Dict[str, Any]:
    """
    Import assets from a module.

    args:
        module_name: name of the module to import from (e.g. 'configz')
        path_parts: path parts to the assets directory (default: [])
        model: model to parse the assets with (default: None)
        load_file: load the file (default: False)
        loader: loader to use (default: None)
        assets_dir: name of the assets directory (default: 'assets')
        allowed_extensions: list of allowed extensions (default: None)
        as_dict: return a dict instead of a list (default: False)
    """
    module_assets = get_module_assets(module_name, *path_parts, assets_dir=assets_dir, allowed_extensions=allowed_extensions, recursive=recursive, **kwargs)
    if not isinstance(module_assets, list):
        module_assets = [module_assets]
    return {asset.name: load_file_content(asset, model=model, loader=loader, **kwargs) if load_file or model is not None else asset for asset in module_assets} if as_dict else [load_file_content(asset, model=model, loader=loader, **kwargs) if load_file or model is not None else asset for asset in module_assets]