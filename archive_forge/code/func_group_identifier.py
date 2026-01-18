from sqlparse import sql
from sqlparse import tokens as T
from sqlparse.utils import recurse, imt
@recurse(sql.Identifier)
def group_identifier(tlist):
    ttypes = (T.String.Symbol, T.Name)
    tidx, token = tlist.token_next_by(t=ttypes)
    while token:
        tlist.group_tokens(sql.Identifier, tidx, tidx)
        tidx, token = tlist.token_next_by(t=ttypes, idx=tidx)