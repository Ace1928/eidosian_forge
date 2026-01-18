import zlib
from typing import Optional, Tuple, Union
from .frame_protocol import CloseReason, FrameDecoder, FrameProtocol, Opcode, RsvBits
def frame_inbound_payload_data(self, proto: Union[FrameDecoder, FrameProtocol], data: bytes) -> Union[bytes, CloseReason]:
    if not self._inbound_compressed or not self._inbound_is_compressible:
        return data
    assert self._decompressor is not None
    try:
        return self._decompressor.decompress(bytes(data))
    except zlib.error:
        return CloseReason.INVALID_FRAME_PAYLOAD_DATA