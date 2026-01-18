from datetime import datetime
import pyparsing as pp
from pyparsing import pyparsing_common as ppc
def convertToDatetime(s, loc, tokens):
    try:
        return datetime(tokens.year, tokens.month, tokens.day).date()
    except Exception as ve:
        errmsg = "'%s/%s/%s' is not a valid date, %s" % (tokens.year, tokens.month, tokens.day, ve)
        raise pp.ParseException(s, loc, errmsg)