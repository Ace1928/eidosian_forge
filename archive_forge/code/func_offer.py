import zlib
from typing import Optional, Tuple, Union
from .frame_protocol import CloseReason, FrameDecoder, FrameProtocol, Opcode, RsvBits
def offer(self) -> Union[bool, str]:
    parameters = ['client_max_window_bits=%d' % self.client_max_window_bits, 'server_max_window_bits=%d' % self.server_max_window_bits]
    if self.client_no_context_takeover:
        parameters.append('client_no_context_takeover')
    if self.server_no_context_takeover:
        parameters.append('server_no_context_takeover')
    return '; '.join(parameters)