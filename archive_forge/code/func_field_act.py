from pyparsing import Literal, Word, delimitedList \
def field_act(s, loc, tok):
    return ('<' + tok[0] + '> ' + ' '.join(tok)).replace('"', '\\"')