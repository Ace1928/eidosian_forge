import sys
from enum import (
from typing import (
def rl_escape_prompt(prompt: str) -> str:
    """Overcome bug in GNU Readline in relation to calculation of prompt length in presence of ANSI escape codes

    :param prompt: original prompt
    :return: prompt safe to pass to GNU Readline
    """
    if rl_type == RlType.GNU:
        escape_start = '\x01'
        escape_end = '\x02'
        escaped = False
        result = ''
        for c in prompt:
            if c == '\x1b' and (not escaped):
                result += escape_start + c
                escaped = True
            elif c.isalpha() and escaped:
                result += c + escape_end
                escaped = False
            else:
                result += c
        return result
    else:
        return prompt