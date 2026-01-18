import threading
from typing import Any, Dict, Optional, Sequence
from uuid import UUID
from langchain_core.callbacks import base as base_callbacks
from langchain_core.documents import Document
from langchain_core.outputs import LLMResult
def on_tool_error(self, error: BaseException, *, run_id: UUID, parent_run_id: Optional[UUID]=None, **kwargs: Any) -> Any:
    if parent_run_id is None:
        self.increment()