import json
import urllib.parse as urllib_parse
def int_parser(s, default_value=0):
    try:
        return int(s)
    except Exception:
        return default_value