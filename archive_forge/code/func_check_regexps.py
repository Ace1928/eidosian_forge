import re
import sys
from optparse import OptionParser
from testtools import ExtendedToStreamDecorator, StreamToExtendedDecorator
from subunit import StreamResultToBytes, read_test_list
from subunit.filters import filter_by_result, find_stream
from subunit.test_results import (TestResultFilter, and_predicates,
def check_regexps(test, outcome, err, details, tags):
    """Check if this test and error match the regexp filters."""
    test_str = str(test) + outcome + str(err) + str(details)
    if with_re and (not with_re.search(test_str)):
        return False
    if without_re and without_re.search(test_str):
        return False
    return True