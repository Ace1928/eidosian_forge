import importlib
import logging
import unicodedata
from codecs import IncrementalDecoder
from encodings.aliases import aliases
from functools import lru_cache
from re import findall
from typing import Generator, List, Optional, Set, Tuple, Union
from _multibytecodec import MultibyteIncrementalDecoder
from .constant import (
def cut_sequence_chunks(sequences: bytes, encoding_iana: str, offsets: range, chunk_size: int, bom_or_sig_available: bool, strip_sig_or_bom: bool, sig_payload: bytes, is_multi_byte_decoder: bool, decoded_payload: Optional[str]=None) -> Generator[str, None, None]:
    if decoded_payload and is_multi_byte_decoder is False:
        for i in offsets:
            chunk = decoded_payload[i:i + chunk_size]
            if not chunk:
                break
            yield chunk
    else:
        for i in offsets:
            chunk_end = i + chunk_size
            if chunk_end > len(sequences) + 8:
                continue
            cut_sequence = sequences[i:i + chunk_size]
            if bom_or_sig_available and strip_sig_or_bom is False:
                cut_sequence = sig_payload + cut_sequence
            chunk = cut_sequence.decode(encoding_iana, errors='ignore' if is_multi_byte_decoder else 'strict')
            if is_multi_byte_decoder and i > 0:
                chunk_partial_size_chk: int = min(chunk_size, 16)
                if decoded_payload and chunk[:chunk_partial_size_chk] not in decoded_payload:
                    for j in range(i, i - 4, -1):
                        cut_sequence = sequences[j:chunk_end]
                        if bom_or_sig_available and strip_sig_or_bom is False:
                            cut_sequence = sig_payload + cut_sequence
                        chunk = cut_sequence.decode(encoding_iana, errors='ignore')
                        if chunk[:chunk_partial_size_chk] in decoded_payload:
                            break
            yield chunk