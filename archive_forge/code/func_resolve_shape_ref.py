from collections import defaultdict
from typing import NamedTuple, Union
from botocore.compat import OrderedDict
from botocore.exceptions import (
from botocore.utils import CachedProperty, hyphenize_service_id, instance_cache
def resolve_shape_ref(self, shape_ref):
    raise ValueError(f"Attempted to resolve shape '{shape_ref}', but no shape map was provided.")