from enum import Enum, IntEnum
from hpack import HeaderTuple
from hyperframe.frame import (
from .errors import ErrorCodes, _error_code_from_int
from .events import (
from .exceptions import (
from .utilities import (
from .windows import WindowManager
def increase_flow_control_window(self, increment):
    """
        Increase the size of the flow control window for the remote side.
        """
    self.config.logger.debug('Increase flow control window for %r by %d', self, increment)
    self.state_machine.process_input(StreamInputs.SEND_WINDOW_UPDATE)
    self._inbound_window_manager.window_opened(increment)
    wuf = WindowUpdateFrame(self.stream_id)
    wuf.window_increment = increment
    return [wuf]