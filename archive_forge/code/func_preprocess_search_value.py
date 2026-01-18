import re
from kombu.utils.encoding import safe_str
def preprocess_search_value(raw_value):
    return raw_value.strip('" ') if raw_value else ''