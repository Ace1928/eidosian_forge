import re
from urllib import parse as urllib_parse
import pyparsing as pp
def cli_to_array(cli_query):
    """Convert CLI list of queries to the Python API format.

    This will convert the following:
        "this<=34;that=string::foo"
    to
        "[{field=this,op=le,value=34,type=''},
          {field=that,op=eq,value=foo,type=string}]"

    """
    opts = []
    queries = cli_query.split(';')
    for q in queries:
        try:
            field, q_operator, type_value = OP_SPLIT_RE.split(q, maxsplit=1)
        except ValueError:
            raise ValueError('Invalid or missing operator in query %(q)s,the supported operators are: %(k)s' % {'q': q, 'k': OP_LOOKUP.keys()})
        if not field:
            raise ValueError('Missing field in query %s' % q)
        if not type_value:
            raise ValueError('Missing value in query %s' % q)
        opt = dict(field=field, op=OP_LOOKUP[q_operator])
        if '::' not in type_value:
            opt['type'], opt['value'] = ('', type_value)
        else:
            opt['type'], _, opt['value'] = type_value.partition('::')
        if opt['type'] and opt['type'] not in ('string', 'integer', 'float', 'datetime', 'boolean'):
            err = 'Invalid value type %(type)s, the type of valueshould be one of: integer, string, float, datetime, boolean.' % opt
            raise ValueError(err)
        opts.append(opt)
    return opts