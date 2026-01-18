import copy
import random
import string
from typing import List, Tuple
import redis
from redis.typing import KeysT, KeyT
def parse_to_dict(response):
    if response is None:
        return {}
    res = {}
    for det in response:
        if isinstance(det[1], list):
            res[det[0]] = parse_list_to_dict(det[1])
        else:
            try:
                try:
                    res[det[0]] = float(det[1])
                except (TypeError, ValueError):
                    res[det[0]] = det[1]
            except IndexError:
                pass
    return res