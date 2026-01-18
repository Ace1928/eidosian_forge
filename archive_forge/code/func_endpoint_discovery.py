from collections import defaultdict
from typing import NamedTuple, Union
from botocore.compat import OrderedDict
from botocore.exceptions import (
from botocore.utils import CachedProperty, hyphenize_service_id, instance_cache
@CachedProperty
def endpoint_discovery(self):
    return self._operation_model.get('endpointdiscovery', None)