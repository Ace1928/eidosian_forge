from __future__ import annotations
from pathlib import Path
from lazyops.types import BaseModel, lazyproperty, validator, Field
from lazyops.configs.base import DefaultSettings, BaseSettings
from typing import List, Dict, Union, Any, Optional
def get_kconfig_path(self, ctx: str=None) -> Path:
    if ctx:
        for ext in {'.yaml', '.yml'}:
            if ext in ctx:
                ext = ''
            path = self.kops_ctx_path.joinpath(f'{ctx}{ext}')
            if path.exists():
                return path
    return self.kconfig_path