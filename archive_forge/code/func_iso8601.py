import json
import argparse
from cliff import command
from datetime import datetime
from iso8601 import iso8601
from iso8601 import ParseError
@staticmethod
def iso8601(argument_value):
    try:
        if argument_value:
            iso8601.parse_date(argument_value)
    except ParseError:
        msg = '%s must be an iso8601 date' % argument_value
        raise argparse.ArgumentTypeError(msg)