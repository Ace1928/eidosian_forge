from collections import defaultdict
from typing import NamedTuple, Union
from botocore.compat import OrderedDict
from botocore.exceptions import (
from botocore.utils import CachedProperty, hyphenize_service_id, instance_cache
@CachedProperty
def event_stream_name(self):
    for member_name, member in self.members.items():
        if member.serialization.get('eventstream'):
            return member_name
    return None