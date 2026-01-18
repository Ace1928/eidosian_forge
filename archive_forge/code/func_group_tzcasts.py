from sqlparse import sql
from sqlparse import tokens as T
from sqlparse.utils import recurse, imt
def group_tzcasts(tlist):

    def match(token):
        return token.ttype == T.Keyword.TZCast

    def valid_prev(token):
        return token is not None

    def valid_next(token):
        return token is not None and (token.is_whitespace or token.match(T.Keyword, 'AS') or token.match(*sql.TypedLiteral.M_CLOSE))

    def post(tlist, pidx, tidx, nidx):
        return (pidx, nidx)
    _group(tlist, sql.Identifier, match, valid_prev, valid_next, post)