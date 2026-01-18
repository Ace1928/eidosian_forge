from sqlparse import sql
from sqlparse import tokens as T
from sqlparse.utils import recurse, imt
@recurse()
def align_comments(tlist):
    tidx, token = tlist.token_next_by(i=sql.Comment)
    while token:
        pidx, prev_ = tlist.token_prev(tidx)
        if isinstance(prev_, sql.TokenList):
            tlist.group_tokens(sql.TokenList, pidx, tidx, extend=True)
            tidx = pidx
        tidx, token = tlist.token_next_by(i=sql.Comment, idx=tidx)