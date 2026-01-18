import re
import sys
import os
from .ansi import AnsiFore, AnsiBack, AnsiStyle, Style, BEL
from .winterm import enable_vt_processing, WinTerm, WinColor, WinStyle
from .win32 import windll, winapi_test
def call_win32(self, command, params):
    if command == 'm':
        for param in params:
            if param in self.win32_calls:
                func_args = self.win32_calls[param]
                func = func_args[0]
                args = func_args[1:]
                kwargs = dict(on_stderr=self.on_stderr)
                func(*args, **kwargs)
    elif command in 'J':
        winterm.erase_screen(params[0], on_stderr=self.on_stderr)
    elif command in 'K':
        winterm.erase_line(params[0], on_stderr=self.on_stderr)
    elif command in 'Hf':
        winterm.set_cursor_position(params, on_stderr=self.on_stderr)
    elif command in 'ABCD':
        n = params[0]
        x, y = {'A': (0, -n), 'B': (0, n), 'C': (n, 0), 'D': (-n, 0)}[command]
        winterm.cursor_adjust(x, y, on_stderr=self.on_stderr)