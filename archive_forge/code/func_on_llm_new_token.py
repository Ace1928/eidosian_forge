import sys
from typing import Any, Dict, List, Optional
from langchain_core.callbacks import StreamingStdOutCallbackHandler
def on_llm_new_token(self, token: str, **kwargs: Any) -> None:
    """Run on new LLM token. Only available when streaming is enabled."""
    self.append_to_last_tokens(token)
    if self.check_if_answer_reached():
        self.answer_reached = True
        if self.stream_prefix:
            for t in self.last_tokens:
                sys.stdout.write(t)
            sys.stdout.flush()
        return
    if self.answer_reached:
        sys.stdout.write(token)
        sys.stdout.flush()