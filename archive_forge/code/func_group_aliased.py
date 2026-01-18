from sqlparse import sql
from sqlparse import tokens as T
from sqlparse.utils import recurse, imt
@recurse()
def group_aliased(tlist):
    I_ALIAS = (sql.Parenthesis, sql.Function, sql.Case, sql.Identifier, sql.Operation, sql.Comparison)
    tidx, token = tlist.token_next_by(i=I_ALIAS, t=T.Number)
    while token:
        nidx, next_ = tlist.token_next(tidx)
        if isinstance(next_, sql.Identifier):
            tlist.group_tokens(sql.Identifier, tidx, nidx, extend=True)
        tidx, token = tlist.token_next_by(i=I_ALIAS, t=T.Number, idx=tidx)