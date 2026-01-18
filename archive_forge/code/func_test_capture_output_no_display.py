import sys
import pytest
from IPython.utils import capture
def test_capture_output_no_display():
    """test capture_output(display=False)"""
    rich = capture.RichOutput(data=full_data)
    with capture.capture_output(display=False) as cap:
        print(hello_stdout, end='')
        print(hello_stderr, end='', file=sys.stderr)
        rich.display()
    assert hello_stdout == cap.stdout
    assert hello_stderr == cap.stderr
    assert cap.outputs == []