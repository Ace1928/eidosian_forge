import os
import sys
def term_title(title):
    term = os.environ.get('TERM', '')
    if term.startswith('xterm') or term == 'dtterm':
        return '\x1b]0;%s\x07' % title
    return ''