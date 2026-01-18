from collections import defaultdict
from typing import NamedTuple, Union
from botocore.compat import OrderedDict
from botocore.exceptions import (
from botocore.utils import CachedProperty, hyphenize_service_id, instance_cache
@CachedProperty
def context_parameters(self):
    if not self.input_shape:
        return []
    return [ContextParameter(name=shape.metadata['contextParam']['name'], member_name=name) for name, shape in self.input_shape.members.items() if 'contextParam' in shape.metadata and 'name' in shape.metadata['contextParam']]