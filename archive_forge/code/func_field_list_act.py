from pyparsing import Literal, Word, delimitedList \
def field_list_act(toks):
    return ' | '.join(toks)