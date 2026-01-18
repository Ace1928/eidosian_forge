import json
import os
import readline
import sys
from yaql import __version__ as version
from yaql.language.exceptions import YaqlParsingException
from yaql.language import utils
def print_output(v, context):
    if context['#nativeOutput']:
        print(v)
    else:
        print(json.dumps(v, indent=4, ensure_ascii=False))