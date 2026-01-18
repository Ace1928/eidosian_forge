from sqlparse import sql
from sqlparse import tokens as T
from sqlparse.utils import recurse, imt
def group_period(tlist):

    def match(token):
        return token.match(T.Punctuation, '.')

    def valid_prev(token):
        sqlcls = (sql.SquareBrackets, sql.Identifier)
        ttypes = (T.Name, T.String.Symbol)
        return imt(token, i=sqlcls, t=ttypes)

    def valid_next(token):
        return True

    def post(tlist, pidx, tidx, nidx):
        sqlcls = (sql.SquareBrackets, sql.Function)
        ttypes = (T.Name, T.String.Symbol, T.Wildcard)
        next_ = tlist[nidx] if nidx is not None else None
        valid_next = imt(next_, i=sqlcls, t=ttypes)
        return (pidx, nidx) if valid_next else (pidx, tidx)
    _group(tlist, sql.Identifier, match, valid_prev, valid_next, post)