import copy
import random
import string
from typing import List, Tuple
import redis
from redis.typing import KeysT, KeyT
def stringify_param_value(value):
    """
    Turn a parameter value into a string suitable for the params header of
    a Cypher command.
    You may pass any value that would be accepted by `json.dumps()`.

    Ways in which output differs from that of `str()`:
        * Strings are quoted.
        * None --> "null".
        * In dictionaries, keys are _not_ quoted.

    :param value: The parameter value to be turned into a string.
    :return: string
    """
    if isinstance(value, str):
        return quote_string(value)
    elif value is None:
        return 'null'
    elif isinstance(value, (list, tuple)):
        return f'[{','.join(map(stringify_param_value, value))}]'
    elif isinstance(value, dict):
        return f'{{{','.join((f'{k}:{stringify_param_value(v)}' for k, v in value.items()))}}}'
    else:
        return str(value)