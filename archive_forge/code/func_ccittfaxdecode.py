import array
from typing import (
def ccittfaxdecode(data: bytes, params: Dict[str, object]) -> bytes:
    K = params.get('K')
    if K == -1:
        cols = cast(int, params.get('Columns'))
        bytealign = cast(bool, params.get('EncodedByteAlign'))
        reversed = cast(bool, params.get('BlackIs1'))
        parser = CCITTFaxDecoder(cols, bytealign=bytealign, reversed=reversed)
    else:
        raise ValueError(K)
    parser.feedbytes(data)
    return parser.close()