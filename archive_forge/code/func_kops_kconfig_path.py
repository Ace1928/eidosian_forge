from __future__ import annotations
from pathlib import Path
from lazyops.types import BaseModel, lazyproperty, validator, Field
from lazyops.configs.base import DefaultSettings, BaseSettings
from typing import List, Dict, Union, Any, Optional
@lazyproperty
def kops_kconfig_path(self):
    if self.kops_ctx and self.kops_ctx_path:
        for ext in {'.yaml', '.yml'}:
            if ext in self.kops_ctx:
                ext = ''
            path = self.kops_ctx_path.joinpath(f'{self.kops_ctx}{ext}')
            if path.exists():
                return path
    return None