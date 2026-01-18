import re
from urllib import parse as urllib_parse
import pyparsing as pp
def format_dict_list(objs, field):
    objs[field] = '\n'.join(('- ' + ', '.join(('%s: %s' % (k, v) for k, v in elem.items())) for elem in objs[field]))