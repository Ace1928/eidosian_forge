import hashlib
from pathlib import Path
from typing import Any, Dict, Iterable, Tuple, Union
def get_custom_callback_meta(self) -> Dict[str, Any]:
    return {'step': self.step, 'starts': self.starts, 'ends': self.ends, 'errors': self.errors, 'text_ctr': self.text_ctr, 'chain_starts': self.chain_starts, 'chain_ends': self.chain_ends, 'llm_starts': self.llm_starts, 'llm_ends': self.llm_ends, 'llm_streams': self.llm_streams, 'tool_starts': self.tool_starts, 'tool_ends': self.tool_ends, 'agent_ends': self.agent_ends}