from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import collections
from googlecloudsdk.command_lib.util.resource_map.declarative import declarative_map
from googlecloudsdk.core import exceptions
def find_krmkinds_by_kind(self, kind):
    """Gets a list of KrmKind keys based on krm kind values."""
    return [x for x in self.krm_map.keys() if x.krm_kind == kind]