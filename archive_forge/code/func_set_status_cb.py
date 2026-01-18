import datetime
import sys
from functools import partial
from optparse import OptionGroup, OptionParser, OptionValueError
from subunit import make_stream_binary
from iso8601 import UTC
from subunit.v2 import StreamResultToBytes
def set_status_cb(option, opt_str, value, parser, status_name):
    if getattr(parser.values, 'action', None) is not None:
        raise OptionValueError('argument %s: Only one status may be specified at once.' % opt_str)
    if len(parser.rargs) == 0:
        raise OptionValueError('argument %s: must specify a single TEST_ID.' % opt_str)
    parser.values.action = status_name
    parser.values.test_id = parser.rargs.pop(0)