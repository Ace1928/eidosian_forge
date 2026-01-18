import sys
from enum import (
from typing import (
def rl_unescape_prompt(prompt: str) -> str:
    """Remove escape characters from a Readline prompt"""
    if rl_type == RlType.GNU:
        escape_start = '\x01'
        escape_end = '\x02'
        prompt = prompt.replace(escape_start, '').replace(escape_end, '')
    return prompt