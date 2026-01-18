from __future__ import annotations
import base64
import hashlib
import json
import os
import tempfile
import uuid
from typing import Optional
import requests
def data_gym_to_mergeable_bpe_ranks(vocab_bpe_file: str, encoder_json_file: str, vocab_bpe_hash: Optional[str]=None, encoder_json_hash: Optional[str]=None) -> dict[bytes, int]:
    rank_to_intbyte = [b for b in range(2 ** 8) if chr(b).isprintable() and chr(b) != ' ']
    data_gym_byte_to_byte = {chr(b): b for b in rank_to_intbyte}
    n = 0
    for b in range(2 ** 8):
        if b not in rank_to_intbyte:
            rank_to_intbyte.append(b)
            data_gym_byte_to_byte[chr(2 ** 8 + n)] = b
            n += 1
    assert len(rank_to_intbyte) == 2 ** 8
    vocab_bpe_contents = read_file_cached(vocab_bpe_file, vocab_bpe_hash).decode()
    bpe_merges = [tuple(merge_str.split()) for merge_str in vocab_bpe_contents.split('\n')[1:-1]]

    def decode_data_gym(value: str) -> bytes:
        return bytes((data_gym_byte_to_byte[b] for b in value))
    bpe_ranks = {bytes([b]): i for i, b in enumerate(rank_to_intbyte)}
    n = len(bpe_ranks)
    for first, second in bpe_merges:
        bpe_ranks[decode_data_gym(first) + decode_data_gym(second)] = n
        n += 1
    encoder_json = json.loads(read_file_cached(encoder_json_file, encoder_json_hash))
    encoder_json_loaded = {decode_data_gym(k): v for k, v in encoder_json.items()}
    encoder_json_loaded.pop(b'<|endoftext|>', None)
    encoder_json_loaded.pop(b'<|startoftext|>', None)
    assert bpe_ranks == encoder_json_loaded
    return bpe_ranks