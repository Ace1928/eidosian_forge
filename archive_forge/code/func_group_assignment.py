from sqlparse import sql
from sqlparse import tokens as T
from sqlparse.utils import recurse, imt
def group_assignment(tlist):

    def match(token):
        return token.match(T.Assignment, ':=')

    def valid(token):
        return token is not None and token.ttype not in T.Keyword

    def post(tlist, pidx, tidx, nidx):
        m_semicolon = (T.Punctuation, ';')
        snidx, _ = tlist.token_next_by(m=m_semicolon, idx=nidx)
        nidx = snidx or nidx
        return (pidx, nidx)
    valid_prev = valid_next = valid
    _group(tlist, sql.Assignment, match, valid_prev, valid_next, post)