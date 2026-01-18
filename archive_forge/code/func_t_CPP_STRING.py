import sys
import re
import copy
import time
import os.path
def t_CPP_STRING(t):
    '''\\"([^\\\\\\n]|(\\\\(.|\\n)))*?\\"'''
    t.lexer.lineno += t.value.count('\n')
    return t