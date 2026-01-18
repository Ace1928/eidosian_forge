from enum import Enum, IntEnum
from hpack import HeaderTuple
from hyperframe.frame import (
from .errors import ErrorCodes, _error_code_from_int
from .events import (
from .exceptions import (
from .utilities import (
from .windows import WindowManager
def advertise_alternative_service(self, field_value):
    """
        Advertise an RFC 7838 alternative service. The semantics of this are
        better documented in the ``H2Connection`` class.
        """
    self.config.logger.debug('Advertise alternative service of %r for %r', field_value, self)
    self.state_machine.process_input(StreamInputs.SEND_ALTERNATIVE_SERVICE)
    asf = AltSvcFrame(self.stream_id)
    asf.field = field_value
    return [asf]