import sys
from typing import Any, Dict, List, Optional
from langchain_core.callbacks import StreamingStdOutCallbackHandler
def on_llm_start(self, serialized: Dict[str, Any], prompts: List[str], **kwargs: Any) -> None:
    """Run when LLM starts running."""
    self.answer_reached = False