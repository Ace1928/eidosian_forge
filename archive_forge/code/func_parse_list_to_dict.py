import copy
import random
import string
from typing import List, Tuple
import redis
from redis.typing import KeysT, KeyT
def parse_list_to_dict(response):
    res = {}
    for i in range(0, len(response), 2):
        if isinstance(response[i], list):
            res['Child iterators'].append(parse_list_to_dict(response[i]))
            try:
                if isinstance(response[i + 1], list):
                    res['Child iterators'].append(parse_list_to_dict(response[i + 1]))
            except IndexError:
                pass
        elif isinstance(response[i + 1], list):
            res['Child iterators'] = [parse_list_to_dict(response[i + 1])]
        else:
            try:
                res[response[i]] = float(response[i + 1])
            except (TypeError, ValueError):
                res[response[i]] = response[i + 1]
    return res