from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import re
from googlecloudsdk.calliope import parser_errors
from googlecloudsdk.core import properties
from googlecloudsdk.core.cache import resource_cache
from googlecloudsdk.core.resource import resource_property
@property
def parsed_args(self):
    return self._parsed_args