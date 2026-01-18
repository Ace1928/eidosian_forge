import re
import sys
import os
from .ansi import AnsiFore, AnsiBack, AnsiStyle, Style, BEL
from .winterm import enable_vt_processing, WinTerm, WinColor, WinStyle
from .win32 import windll, winapi_test
def convert_ansi(self, paramstring, command):
    if self.convert:
        params = self.extract_params(command, paramstring)
        self.call_win32(command, params)