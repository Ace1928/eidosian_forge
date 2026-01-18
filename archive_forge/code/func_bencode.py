from typing import Dict, Type, Callable, List
def bencode(x):
    r = []
    encode_func[type(x)](x, r)
    return b''.join(r)