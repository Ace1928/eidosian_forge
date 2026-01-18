import sys
from typing import Any, Dict, List, Optional
from langchain_core.callbacks import StreamingStdOutCallbackHandler
def append_to_last_tokens(self, token: str) -> None:
    self.last_tokens.append(token)
    self.last_tokens_stripped.append(token.strip())
    if len(self.last_tokens) > len(self.answer_prefix_tokens):
        self.last_tokens.pop(0)
        self.last_tokens_stripped.pop(0)