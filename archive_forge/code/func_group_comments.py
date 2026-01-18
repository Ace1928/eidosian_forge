from sqlparse import sql
from sqlparse import tokens as T
from sqlparse.utils import recurse, imt
@recurse(sql.Comment)
def group_comments(tlist):
    tidx, token = tlist.token_next_by(t=T.Comment)
    while token:
        eidx, end = tlist.token_not_matching(lambda tk: imt(tk, t=T.Comment) or tk.is_whitespace, idx=tidx)
        if end is not None:
            eidx, end = tlist.token_prev(eidx, skip_ws=False)
            tlist.group_tokens(sql.Comment, tidx, eidx)
        tidx, token = tlist.token_next_by(t=T.Comment, idx=tidx)