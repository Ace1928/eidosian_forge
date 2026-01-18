from typing import Dict, Type, Callable, List
def encode_int(x, r):
    r.extend((b'i', int_to_bytes(x), b'e'))