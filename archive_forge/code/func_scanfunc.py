from typing import List, Union
from ray.rllib.utils.framework import try_import_tf
def scanfunc(acc, sequence_item):
    discount_t, c_t, delta_t = sequence_item
    return delta_t + discount_t * c_t * acc