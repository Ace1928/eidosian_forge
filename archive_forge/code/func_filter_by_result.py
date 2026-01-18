import sys
from optparse import OptionParser
from testtools import CopyStreamResult, StreamResult, StreamResultRouter
from subunit import (ByteStreamToStreamResult, DiscardStream, ProtocolTestCase,
from subunit.test_results import CatFiles
def filter_by_result(result_factory, output_path, passthrough, forward, input_stream=sys.stdin, protocol_version=1, passthrough_subunit=True):
    """Filter an input stream using a test result.

    :param result_factory: A callable that when passed an output stream
        returns a TestResult.  It is expected that this result will output
        to the given stream.
    :param output_path: A path send output to.  If None, output will be go
        to ``sys.stdout``.
    :param passthrough: If True, all non-subunit input will be sent to
        ``sys.stdout``.  If False, that input will be discarded.
    :param forward: If True, all subunit input will be forwarded directly to
        ``sys.stdout`` as well as to the ``TestResult``.
    :param input_stream: The source of subunit input.  Defaults to
        ``sys.stdin``.
    :param protocol_version: The subunit protocol version to expect.
    :param passthrough_subunit: If True, passthrough should be as subunit.
    :return: A test result with the results of the run.
    """
    if passthrough:
        passthrough_stream = sys.stdout
    elif 1 == protocol_version:
        passthrough_stream = DiscardStream()
    else:
        passthrough_stream = None
    if forward:
        forward_stream = sys.stdout
    elif 1 == protocol_version:
        forward_stream = DiscardStream()
    else:
        forward_stream = None
    if output_path is None:
        output_to = sys.stdout
    else:
        output_to = open(output_path, 'w')
    try:
        result = result_factory(output_to)
        run_tests_from_stream(input_stream, result, passthrough_stream, forward_stream, protocol_version=protocol_version, passthrough_subunit=passthrough_subunit)
    finally:
        if output_path:
            output_to.close()
    return result