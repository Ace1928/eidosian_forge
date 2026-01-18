from sqlparse import sql
from sqlparse import tokens as T
from sqlparse.utils import recurse, imt
def group_identifier_list(tlist):
    m_role = (T.Keyword, ('null', 'role'))
    sqlcls = (sql.Function, sql.Case, sql.Identifier, sql.Comparison, sql.IdentifierList, sql.Operation)
    ttypes = T_NUMERICAL + T_STRING + T_NAME + (T.Keyword, T.Comment, T.Wildcard)

    def match(token):
        return token.match(T.Punctuation, ',')

    def valid(token):
        return imt(token, i=sqlcls, m=m_role, t=ttypes)

    def post(tlist, pidx, tidx, nidx):
        return (pidx, nidx)
    valid_prev = valid_next = valid
    _group(tlist, sql.IdentifierList, match, valid_prev, valid_next, post, extend=True)