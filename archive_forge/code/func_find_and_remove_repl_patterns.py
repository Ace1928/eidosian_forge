import os
import sys
import re
def find_and_remove_repl_patterns(astr):
    names = find_repl_patterns(astr)
    astr = re.subn(named_re, '', astr)[0]
    return (astr, names)