import typing
import functools
import pathlib
import importlib.util
from lazyops.utils.lazy import lazy_import
from typing import Optional, Dict, List, Union, Any, Callable, Type, TYPE_CHECKING
@functools.lru_cache()
def get_module_assets_path(module_name: str, assets_dir: typing.Optional[str]='assets', **kwargs) -> pathlib.Path:
    """
    Get the path to the module's assets.

    args:
        module_name: name of the module to import from (e.g. 'lazyops')
        assets_dir: name of the assets directory (default: 'assets')
    
    Use it like this:

    >>> get_module_assets_path('lazyops', assets_dir = 'assets')
    """
    module_spec = importlib.util.find_spec(module_name)
    if not module_spec:
        raise ValueError(f'Module {module_name} not found')
    for path in module_spec.submodule_search_locations:
        asset_path = pathlib.Path(path).joinpath(assets_dir)
        if asset_path.exists():
            return asset_path
    raise ValueError(f'Module {module_name} does not have an assets directory')