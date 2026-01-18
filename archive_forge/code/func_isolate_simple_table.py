import sys
import re
from types import FunctionType, MethodType
from docutils import nodes, statemachine, utils
from docutils import ApplicationError, DataError
from docutils.statemachine import StateMachineWS, StateWS
from docutils.nodes import fully_normalize_name as normalize_name
from docutils.nodes import whitespace_normalize_name
import docutils.parsers.rst
from docutils.parsers.rst import directives, languages, tableparser, roles
from docutils.parsers.rst.languages import en as _fallback_language_module
from docutils.utils import escape2null, unescape, column_width
from docutils.utils import punctuation_chars, roman, urischemes
from docutils.utils import split_escaped_whitespace
def isolate_simple_table(self):
    start = self.state_machine.line_offset
    lines = self.state_machine.input_lines
    limit = len(lines) - 1
    toplen = len(lines[start].strip())
    pattern_match = self.simple_table_border_pat.match
    found = 0
    found_at = None
    i = start + 1
    while i <= limit:
        line = lines[i]
        match = pattern_match(line)
        if match:
            if len(line.strip()) != toplen:
                self.state_machine.next_line(i - start)
                messages = self.malformed_table(lines[start:i + 1], 'Bottom/header table border does not match top border.')
                return ([], messages, i == limit or not lines[i + 1].strip())
            found += 1
            found_at = i
            if found == 2 or i == limit or (not lines[i + 1].strip()):
                end = i
                break
        i += 1
    else:
        if found:
            extra = ' or no blank line after table bottom'
            self.state_machine.next_line(found_at - start)
            block = lines[start:found_at + 1]
        else:
            extra = ''
            self.state_machine.next_line(i - start - 1)
            block = lines[start:]
        messages = self.malformed_table(block, 'No bottom table border found%s.' % extra)
        return ([], messages, not extra)
    self.state_machine.next_line(end - start)
    block = lines[start:end + 1]
    block.pad_double_width(self.double_width_pad_char)
    return (block, [], end == limit or not lines[end + 1].strip())