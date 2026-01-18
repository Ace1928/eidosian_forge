from __future__ import annotations
from typing import Any, Callable, Dict, Optional
from langchain_core.output_parsers import BaseOutputParser
@classmethod
def from_rail_string(cls, rail_str: str, num_reasks: int=1, api: Optional[Callable]=None, *args: Any, **kwargs: Any) -> GuardrailsOutputParser:
    try:
        from guardrails import Guard
    except ImportError:
        raise ImportError('guardrails-ai package not installed. Install it by running `pip install guardrails-ai`.')
    return cls(guard=Guard.from_rail_string(rail_str, num_reasks=num_reasks), api=api, args=args, kwargs=kwargs)