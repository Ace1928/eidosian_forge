import re
from urllib import parse as urllib_parse
import pyparsing as pp
def format_move_dict_to_root(obj, field):
    for attr in obj[field]:
        obj['%s/%s' % (field, attr)] = obj[field][attr]
    del obj[field]