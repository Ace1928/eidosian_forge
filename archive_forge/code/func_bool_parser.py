import json
import urllib.parse as urllib_parse
def bool_parser(s):
    return s in ('True', 'true', '1', True, 1)