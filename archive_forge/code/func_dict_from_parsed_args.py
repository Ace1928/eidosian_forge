import re
from urllib import parse as urllib_parse
import pyparsing as pp
def dict_from_parsed_args(parsed_args, attrs):
    d = {}
    for attr in attrs:
        if attr == 'metric':
            if parsed_args.metrics:
                value = parsed_args.metrics[0]
            else:
                value = None
        else:
            value = getattr(parsed_args, attr)
        if value is not None:
            if value == ['']:
                d[attr] = None
            else:
                d[attr] = value
    return d