from __future__ import annotations
from pathlib import Path
from lazyops.types import BaseModel, lazyproperty, validator, Field
from lazyops.configs.base import DefaultSettings, BaseSettings
from typing import List, Dict, Union, Any, Optional
@lazyproperty
def kops_ctx_path(self):
    return Path(self.kops_ctx_dir) if self.kops_ctx_dir else None