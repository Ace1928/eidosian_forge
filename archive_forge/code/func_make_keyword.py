import pyparsing as pp
from pyparsing import pyparsing_common as ppc
def make_keyword(kwd_str, kwd_value):
    return pp.Keyword(kwd_str).setParseAction(pp.replaceWith(kwd_value))