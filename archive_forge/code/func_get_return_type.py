import json
import pydoc
from kubernetes import client
def get_return_type(self, func):
    if self._raw_return_type:
        return self._raw_return_type
    return_type = _find_return_type(func)
    if return_type.endswith(TYPE_LIST_SUFFIX):
        return return_type[:-len(TYPE_LIST_SUFFIX)]
    return return_type