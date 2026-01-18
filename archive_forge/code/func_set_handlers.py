from __future__ import annotations
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Sequence, TypeVar, Union
from uuid import UUID
from tenacity import RetryCallState
def set_handlers(self, handlers: List[BaseCallbackHandler], inherit: bool=True) -> None:
    """Set handlers as the only handlers on the callback manager."""
    self.handlers = []
    self.inheritable_handlers = []
    for handler in handlers:
        self.add_handler(handler, inherit=inherit)