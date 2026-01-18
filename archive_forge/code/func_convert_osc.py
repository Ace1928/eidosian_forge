import re
import sys
import os
from .ansi import AnsiFore, AnsiBack, AnsiStyle, Style, BEL
from .winterm import enable_vt_processing, WinTerm, WinColor, WinStyle
from .win32 import windll, winapi_test
def convert_osc(self, text):
    for match in self.ANSI_OSC_RE.finditer(text):
        start, end = match.span()
        text = text[:start] + text[end:]
        paramstring, command = match.groups()
        if command == BEL:
            if paramstring.count(';') == 1:
                params = paramstring.split(';')
                if params[0] in '02':
                    winterm.set_title(params[1])
    return text