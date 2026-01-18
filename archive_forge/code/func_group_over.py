from sqlparse import sql
from sqlparse import tokens as T
from sqlparse.utils import recurse, imt
@recurse(sql.Over)
def group_over(tlist):
    tidx, token = tlist.token_next_by(m=sql.Over.M_OPEN)
    while token:
        nidx, next_ = tlist.token_next(tidx)
        if imt(next_, i=sql.Parenthesis, t=T.Name):
            tlist.group_tokens(sql.Over, tidx, nidx)
        tidx, token = tlist.token_next_by(m=sql.Over.M_OPEN, idx=tidx)