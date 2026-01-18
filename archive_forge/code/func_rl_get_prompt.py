import sys
from enum import (
from typing import (
def rl_get_prompt() -> str:
    """Gets Readline's current prompt"""
    if rl_type == RlType.GNU:
        encoded_prompt = ctypes.c_char_p.in_dll(readline_lib, 'rl_prompt').value
        if encoded_prompt is None:
            prompt = ''
        else:
            prompt = encoded_prompt.decode(encoding='utf-8')
    elif rl_type == RlType.PYREADLINE:
        prompt_data: Union[str, bytes] = readline.rl.prompt
        if isinstance(prompt_data, bytes):
            prompt = prompt_data.decode(encoding='utf-8')
        else:
            prompt = prompt_data
    else:
        prompt = ''
    return rl_unescape_prompt(prompt)