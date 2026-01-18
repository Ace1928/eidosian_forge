from traitlets import Bool, Tuple, List
from .utils import setup, teardown, DummyComm
from ..widget import Widget
from ..._version import __control_protocol_version__
def test_empty_hold_sync():
    w = SimpleWidget()
    with w.hold_sync():
        pass
    assert w.comm.messages == []