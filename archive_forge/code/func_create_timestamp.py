import datetime
import sys
from functools import partial
from optparse import OptionGroup, OptionParser, OptionValueError
from subunit import make_stream_binary
from iso8601 import UTC
from subunit.v2 import StreamResultToBytes
def create_timestamp():
    return datetime.datetime.now(UTC)