from sqlparse import sql
from sqlparse import tokens as T
from sqlparse.utils import recurse, imt
def group_if(tlist):
    _group_matching(tlist, sql.If)