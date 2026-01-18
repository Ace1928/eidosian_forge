from collections import defaultdict
from typing import NamedTuple, Union
from botocore.compat import OrderedDict
from botocore.exceptions import (
from botocore.utils import CachedProperty, hyphenize_service_id, instance_cache
@CachedProperty
def static_context_parameters(self):
    params = self._operation_model.get('staticContextParams', {})
    return [StaticContextParameter(name=name, value=props.get('value')) for name, props in params.items()]