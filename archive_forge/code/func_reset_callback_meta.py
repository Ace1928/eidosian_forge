import hashlib
from pathlib import Path
from typing import Any, Dict, Iterable, Tuple, Union
def reset_callback_meta(self) -> None:
    """Reset the callback metadata."""
    self.step = 0
    self.starts = 0
    self.ends = 0
    self.errors = 0
    self.text_ctr = 0
    self.ignore_llm_ = False
    self.ignore_chain_ = False
    self.ignore_agent_ = False
    self.always_verbose_ = False
    self.chain_starts = 0
    self.chain_ends = 0
    self.llm_starts = 0
    self.llm_ends = 0
    self.llm_streams = 0
    self.tool_starts = 0
    self.tool_ends = 0
    self.agent_ends = 0
    self.on_llm_start_records = []
    self.on_llm_token_records = []
    self.on_llm_end_records = []
    self.on_chain_start_records = []
    self.on_chain_end_records = []
    self.on_tool_start_records = []
    self.on_tool_end_records = []
    self.on_text_records = []
    self.on_agent_finish_records = []
    self.on_agent_action_records = []
    return None