from __future__ import annotations
from lazyops.utils.lazy import lazy_import
from typing import Any, Dict, Optional, Type, Union, TYPE_CHECKING
def get_module_settings(module: Optional[str]=None, settings_cls: Optional[Union[Type['AppSettings'], str]]=None) -> 'AppSettings':
    """
    Get the module settings
    """
    if module is None:
        if not _lazy_module_settings:
            raise ValueError('No module settings registered')
        module = list(_lazy_module_settings.keys())[-1]
    if module not in _lazy_module_settings:
        if settings_cls is None:
            try:
                settings_cls = lazy_import(f'{module}.configs.settings')
            except ImportError as e:
                raise ValueError(f'Module settings for {module} not found') from e
        elif isinstance(settings_cls, str):
            settings_cls = lazy_import(settings_cls)
        if isinstance(settings_cls, type):
            settings_cls = settings_cls()
        _lazy_module_settings[module] = settings_cls
    return _lazy_module_settings[module]