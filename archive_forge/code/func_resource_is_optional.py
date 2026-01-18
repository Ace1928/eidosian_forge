from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import itertools
import enum
@property
def resource_is_optional(self):
    return self.resource_type in TriggerEvent.OPTIONAL_RESOURCE_TYPES