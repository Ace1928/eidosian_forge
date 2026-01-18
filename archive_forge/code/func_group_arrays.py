from sqlparse import sql
from sqlparse import tokens as T
from sqlparse.utils import recurse, imt
def group_arrays(tlist):
    sqlcls = (sql.SquareBrackets, sql.Identifier, sql.Function)
    ttypes = (T.Name, T.String.Symbol)

    def match(token):
        return isinstance(token, sql.SquareBrackets)

    def valid_prev(token):
        return imt(token, i=sqlcls, t=ttypes)

    def valid_next(token):
        return True

    def post(tlist, pidx, tidx, nidx):
        return (pidx, tidx)
    _group(tlist, sql.Identifier, match, valid_prev, valid_next, post, extend=True, recurse=False)