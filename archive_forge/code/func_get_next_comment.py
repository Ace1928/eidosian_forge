from sqlparse import sql, tokens as T
from sqlparse.utils import split_unquoted_newlines
def get_next_comment():
    return tlist.token_next_by(i=sql.Comment, t=T.Comment)