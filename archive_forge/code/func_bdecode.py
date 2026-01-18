from typing import Dict, Type, Callable, List
def bdecode(self, x):
    if not isinstance(x, bytes):
        raise TypeError
    try:
        r, l = self.decode_func[x[:1]](x, 0)
    except (IndexError, KeyError, OverflowError) as e:
        raise ValueError(str(e))
    if l != len(x):
        raise ValueError
    return r