import errno as errno_mod
import zmq
from zmq.constants import SocketOption, _OptType
from zmq.error import ZMQError, _check_rc, _check_version
from ._cffi import ffi
from ._cffi import lib as C
from .message import Frame
from .utils import _retry_sys_call
def new_uint64_pointer():
    return (ffi.new('uint64_t*'), nsp(ffi.sizeof('uint64_t')))