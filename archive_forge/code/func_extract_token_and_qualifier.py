from collections import namedtuple
from string import ascii_letters, digits
from _pydevd_bundle import pydevd_xml
import pydevconsole
import builtins as __builtin__  # Py3
def extract_token_and_qualifier(text, line=0, column=0):
    """
    Extracts the token a qualifier from the text given the line/colum
    (see test_extract_token_and_qualifier for examples).

    :param unicode text:
    :param int line: 0-based
    :param int column: 0-based
    """
    if line < 0:
        line = 0
    if column < 0:
        column = 0
    if isinstance(text, bytes):
        text = text.decode('utf-8')
    lines = text.splitlines()
    try:
        text = lines[line]
    except IndexError:
        return TokenAndQualifier(u'', u'')
    if column >= len(text):
        column = len(text)
    text = text[:column]
    token = u''
    qualifier = u''
    temp_token = []
    for i in range(column - 1, -1, -1):
        c = text[i]
        if c in identifier_part or isidentifier(c) or c == u'.':
            temp_token.append(c)
        else:
            break
    temp_token = u''.join(reversed(temp_token))
    if u'.' in temp_token:
        temp_token = temp_token.split(u'.')
        token = u'.'.join(temp_token[:-1])
        qualifier = temp_token[-1]
    else:
        qualifier = temp_token
    return TokenAndQualifier(token, qualifier)