from collections import defaultdict
from typing import NamedTuple, Union
from botocore.compat import OrderedDict
from botocore.exceptions import (
from botocore.utils import CachedProperty, hyphenize_service_id, instance_cache
@instance_cache
def operation_model(self, operation_name):
    try:
        model = self._service_description['operations'][operation_name]
    except KeyError:
        raise OperationNotFoundError(operation_name)
    return OperationModel(model, self, operation_name)