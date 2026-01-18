import sys
from optparse import OptionParser
from testtools import StreamResultRouter, StreamToExtendedDecorator
from subunit import ByteStreamToStreamResult, TestProtocolClient
from subunit.filters import find_stream
from subunit.test_results import CatFiles
def make_options(description):
    parser = OptionParser(description=__doc__)
    return parser