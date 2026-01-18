from __future__ import annotations
import abc
import contextlib
from kvdb.io import cachify as _cachify
from typing import Optional, Type, TypeVar, Union, Set, List, Any, Dict, Literal, TYPE_CHECKING
def cachify_create_base_name(self, func: str, **kwargs) -> str:
    """
        Creates the base name
        """
    return f'{self.settings.ctx.module_name}.{self.name}' if self.cachify_shared_global else f'{self.settings.ctx.module_name}.{self.settings.app_env.name}.{self.name}'