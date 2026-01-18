from collections import defaultdict
from typing import NamedTuple, Union
from botocore.compat import OrderedDict
from botocore.exceptions import (
from botocore.utils import CachedProperty, hyphenize_service_id, instance_cache
@CachedProperty
def endpoint_discovery_required(self):
    for operation in self.operation_names:
        model = self.operation_model(operation)
        if model.endpoint_discovery is not None and model.endpoint_discovery.get('required'):
            return True
    return False