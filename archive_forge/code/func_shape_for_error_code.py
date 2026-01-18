from collections import defaultdict
from typing import NamedTuple, Union
from botocore.compat import OrderedDict
from botocore.exceptions import (
from botocore.utils import CachedProperty, hyphenize_service_id, instance_cache
def shape_for_error_code(self, error_code):
    return self._error_code_cache.get(error_code, None)