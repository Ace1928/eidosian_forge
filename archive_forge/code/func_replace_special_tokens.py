import csv
import functools
import logging
import re
from importlib import import_module
from humanfriendly.compat import StringIO
from humanfriendly.text import dedent, split_paragraphs, trim_empty_lines
def replace_special_tokens(text, meta_variables, replace_fn):
    return USAGE_PATTERN.sub(functools.partial(replace_tokens_callback, meta_variables=meta_variables, replace_fn=replace_fn), text)