import re
from pygments.lexers import (
from pygments.lexer import (
from pygments.token import (
from pygments.util import get_bool_opt
def get_mci(self, line):
    """
        Parses the line and returns a 3-tuple: (mode, code, insertion).

        `mode` is the next mode (or state) of the lexer, and is always equal
        to 'input', 'output', or 'tb'.

        `code` is a portion of the line that should be added to the buffer
        corresponding to the next mode and eventually lexed by another lexer.
        For example, `code` could be Python code if `mode` were 'input'.

        `insertion` is a 3-tuple (index, token, text) representing an
        unprocessed "token" that will be inserted into the stream of tokens
        that are created from the buffer once we change modes. This is usually
        the input or output prompt.

        In general, the next mode depends on current mode and on the contents
        of `line`.

        """
    in2_match = self.in2_regex.match(line)
    in2_match_rstrip = self.in2_regex_rstrip.match(line)
    if in2_match and in2_match.group().rstrip() == line.rstrip() or in2_match_rstrip:
        end_input = True
    else:
        end_input = False
    if end_input and self.mode != 'tb':
        mode = 'output'
        code = u''
        insertion = (0, Generic.Prompt, line)
        return (mode, code, insertion)
    out_match = self.out_regex.match(line)
    out_match_rstrip = self.out_regex_rstrip.match(line)
    if out_match or out_match_rstrip:
        mode = 'output'
        if out_match:
            idx = out_match.end()
        else:
            idx = out_match_rstrip.end()
        code = line[idx:]
        insertion = (0, Generic.Heading, line[:idx])
        return (mode, code, insertion)
    in1_match = self.in1_regex.match(line)
    if in1_match or (in2_match and self.mode != 'tb'):
        mode = 'input'
        if in1_match:
            idx = in1_match.end()
        else:
            idx = in2_match.end()
        code = line[idx:]
        insertion = (0, Generic.Prompt, line[:idx])
        return (mode, code, insertion)
    in1_match_rstrip = self.in1_regex_rstrip.match(line)
    if in1_match_rstrip or (in2_match_rstrip and self.mode != 'tb'):
        mode = 'input'
        if in1_match_rstrip:
            idx = in1_match_rstrip.end()
        else:
            idx = in2_match_rstrip.end()
        code = line[idx:]
        insertion = (0, Generic.Prompt, line[:idx])
        return (mode, code, insertion)
    if self.ipytb_start.match(line):
        mode = 'tb'
        code = line
        insertion = None
        return (mode, code, insertion)
    if self.mode in ('input', 'output'):
        mode = 'output'
    else:
        mode = 'tb'
    code = line
    insertion = None
    return (mode, code, insertion)