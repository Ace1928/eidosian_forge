from datetime import datetime, time, timedelta
import pyparsing as pp
import calendar
from_ = CK("from").setParseAction(pp.replaceWith(1))
def make_integer_word_expr(int_name, int_value):
    return pp.CaselessKeyword(int_name).addParseAction(pp.replaceWith(int_value))