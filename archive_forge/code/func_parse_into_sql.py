from parlai.core.teachers import DialogTeacher
from .build import build
import os
import json
def parse_into_sql(table, query):
    header = table['header']
    sql_query = 'SELECT {agg} {sel} FROM table'.format(agg=self.agg_ops[query['agg']], sel=header[query['sel']])
    if query['conds']:
        sql_query += ' WHERE ' + ' AND '.join(['{} {} {}'.format(header[i], self.cond_ops[o], v) for i, o, v in query['conds']])
    return sql_query