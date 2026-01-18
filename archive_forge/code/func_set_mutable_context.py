from __future__ import annotations
from pydantic import model_validator
from lazyops.types import BaseModel, Field
from typing import Optional, Dict, Any, Union, List
@model_validator(mode='after')
def set_mutable_context(self):
    """
        Sets the ctext
        """
    if self.text is not None:
        self.ctext = self.text
    if self.command is not None:
        self.ccommand = self.command
    return self