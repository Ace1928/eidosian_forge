from typing import Dict, Type, Callable, List
def decode_dict(self, x, f):
    r, f = ({}, f + 1)
    lastkey = None
    while x[f:f + 1] != b'e':
        k, f = self.decode_string(x, f)
        if lastkey is not None and lastkey >= k:
            raise ValueError
        lastkey = k
        r[k], f = self.decode_func[x[f:f + 1]](x, f)
    return (r, f + 1)