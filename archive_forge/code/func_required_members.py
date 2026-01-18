from collections import defaultdict
from typing import NamedTuple, Union
from botocore.compat import OrderedDict
from botocore.exceptions import (
from botocore.utils import CachedProperty, hyphenize_service_id, instance_cache
@CachedProperty
def required_members(self):
    """A list of members that are required.

        A structure shape can define members that are required.
        This value will return a list of required members.  If there
        are no required members an empty list is returned.

        """
    return self.metadata.get('required', [])