import ast
import os
import re
from hacking import core
from os_win.utils.winapi import libs as w_lib
import_translation_for_log_or_exception = re.compile(
@core.flake8ext
def capital_cfg_help(logical_line, tokens):
    msg = 'N313: capitalize help string'
    if cfg_re.match(logical_line):
        for t in range(len(tokens)):
            if tokens[t][1] == 'help':
                txt = tokens[t + 2][1]
                if len(txt) > 1 and txt[1].islower():
                    yield (0, msg)