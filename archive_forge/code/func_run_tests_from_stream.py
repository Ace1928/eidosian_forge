import sys
from optparse import OptionParser
from testtools import CopyStreamResult, StreamResult, StreamResultRouter
from subunit import (ByteStreamToStreamResult, DiscardStream, ProtocolTestCase,
from subunit.test_results import CatFiles
def run_tests_from_stream(input_stream, result, passthrough_stream=None, forward_stream=None, protocol_version=1, passthrough_subunit=True):
    """Run tests from a subunit input stream through 'result'.

    Non-test events - top level file attachments - are expected to be
    dropped by v2 StreamResults at the present time (as all the analysis code
    is in ExtendedTestResult API's), so to implement passthrough_stream they
    are diverted and copied directly when that is set.

    :param input_stream: A stream containing subunit input.
    :param result: A TestResult that will receive the test events.
        NB: This should be an ExtendedTestResult for v1 and a StreamResult for
        v2.
    :param passthrough_stream: All non-subunit input received will be
        sent to this stream.  If not provided, uses the ``TestProtocolServer``
        default, which is ``sys.stdout``.
    :param forward_stream: All subunit input received will be forwarded
        to this stream. If not provided, uses the ``TestProtocolServer``
        default, which is to not forward any input. Do not set this when
        transforming the stream - items would be double-reported.
    :param protocol_version: What version of the subunit protocol to expect.
    :param passthrough_subunit: If True, passthrough should be as subunit
        otherwise unwrap it. Only has effect when forward_stream is None.
        (when forwarding as subunit non-subunit input is always turned into
        subunit)
    """
    if 1 == protocol_version:
        test = ProtocolTestCase(input_stream, passthrough=passthrough_stream, forward=forward_stream)
    elif 2 == protocol_version:
        if forward_stream is not None:
            forward_result = StreamResultToBytes(forward_stream)
            if passthrough_stream is None:
                router = StreamResultRouter(forward_result)
                router.add_rule(StreamResult(), 'test_id', test_id=None)
                result = CopyStreamResult([router, result])
            else:
                result = CopyStreamResult([forward_result, result])
        elif passthrough_stream is not None:
            if not passthrough_subunit:
                passthrough_result = CatFiles(passthrough_stream)
            else:
                passthrough_result = StreamResultToBytes(passthrough_stream)
            result = StreamResultRouter(result)
            result.add_rule(passthrough_result, 'test_id', test_id=None)
        test = ByteStreamToStreamResult(input_stream, non_subunit_name='stdout')
    else:
        raise Exception('Unknown protocol version.')
    result.startTestRun()
    test.run(result)
    result.stopTestRun()