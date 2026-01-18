import sys
from enum import (
from typing import (
def rl_set_prompt(prompt: str) -> None:
    """
    Sets Readline's prompt
    :param prompt: the new prompt value
    """
    escaped_prompt = rl_escape_prompt(prompt)
    if rl_type == RlType.GNU:
        encoded_prompt = bytes(escaped_prompt, encoding='utf-8')
        readline_lib.rl_set_prompt(encoded_prompt)
    elif rl_type == RlType.PYREADLINE:
        readline.rl.prompt = escaped_prompt