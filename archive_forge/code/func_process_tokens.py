import os
import sys
import tokenize
def process_tokens(tokens):
    INDENT = tokenize.INDENT
    DEDENT = tokenize.DEDENT
    NEWLINE = tokenize.NEWLINE
    JUNK = (tokenize.COMMENT, tokenize.NL)
    indents = [Whitespace('')]
    check_equal = 0
    for type, token, start, end, line in tokens:
        if type == NEWLINE:
            check_equal = 1
        elif type == INDENT:
            check_equal = 0
            thisguy = Whitespace(token)
            if not indents[-1].less(thisguy):
                witness = indents[-1].not_less_witness(thisguy)
                msg = 'indent not greater e.g. ' + format_witnesses(witness)
                raise NannyNag(start[0], msg, line)
            indents.append(thisguy)
        elif type == DEDENT:
            check_equal = 1
            del indents[-1]
        elif check_equal and type not in JUNK:
            check_equal = 0
            thisguy = Whitespace(line)
            if not indents[-1].equal(thisguy):
                witness = indents[-1].not_equal_witness(thisguy)
                msg = 'indent not equal e.g. ' + format_witnesses(witness)
                raise NannyNag(start[0], msg, line)