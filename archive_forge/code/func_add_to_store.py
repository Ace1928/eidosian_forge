from functools import wraps
from keras.src.backend.common.global_state import get_global_attribute
from keras.src.backend.common.global_state import set_global_attribute
from keras.src.utils import python_utils
def add_to_store(self, store_name, value):
    if self.locked:
        raise ValueError(self._lock_violation_msg)
    self.config[store_name][1].append(value)
    self.stored_ids[store_name].add(id(value))