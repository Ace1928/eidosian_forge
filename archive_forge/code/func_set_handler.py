from __future__ import annotations
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Sequence, TypeVar, Union
from uuid import UUID
from tenacity import RetryCallState
def set_handler(self, handler: BaseCallbackHandler, inherit: bool=True) -> None:
    """Set handler as the only handler on the callback manager."""
    self.set_handlers([handler], inherit=inherit)