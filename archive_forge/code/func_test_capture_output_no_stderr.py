import sys
import pytest
from IPython.utils import capture
def test_capture_output_no_stderr():
    """test capture_output(stderr=False)"""
    rich = capture.RichOutput(data=full_data)
    with capture.capture_output(), capture.capture_output(stderr=False) as cap:
        print(hello_stdout, end='')
        print(hello_stderr, end='', file=sys.stderr)
        rich.display()
    assert hello_stdout == cap.stdout
    assert '' == cap.stderr
    assert len(cap.outputs) == 1