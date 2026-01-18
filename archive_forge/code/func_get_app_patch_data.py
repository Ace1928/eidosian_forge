from __future__ import annotations
from .base import BaseModel
from pydantic import Field, PrivateAttr
from ..utils.lazy import logger
from typing import Optional, Dict, Any, Union, List, TYPE_CHECKING
def get_app_patch_data(self) -> Dict[str, List[str]]:
    """
        Returns the patch data
        """
    return {'allowed_origins': self.allowed_origins, 'callbacks': self.callbacks, 'web_origins': self.web_origins, 'allowed_logout_urls': self.allowed_logout_urls}