from sqlparse import sql
from sqlparse import tokens as T
from sqlparse.utils import recurse, imt
def match_to_extend(token):
    return isinstance(token, sql.TypedLiteral)