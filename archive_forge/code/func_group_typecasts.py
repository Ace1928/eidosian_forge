from sqlparse import sql
from sqlparse import tokens as T
from sqlparse.utils import recurse, imt
def group_typecasts(tlist):

    def match(token):
        return token.match(T.Punctuation, '::')

    def valid(token):
        return token is not None

    def post(tlist, pidx, tidx, nidx):
        return (pidx, nidx)
    valid_prev = valid_next = valid
    _group(tlist, sql.Identifier, match, valid_prev, valid_next, post)