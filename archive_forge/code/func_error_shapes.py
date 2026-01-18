from collections import defaultdict
from typing import NamedTuple, Union
from botocore.compat import OrderedDict
from botocore.exceptions import (
from botocore.utils import CachedProperty, hyphenize_service_id, instance_cache
@CachedProperty
def error_shapes(self):
    shapes = self._operation_model.get('errors', [])
    return list((self._service_model.resolve_shape_ref(s) for s in shapes))