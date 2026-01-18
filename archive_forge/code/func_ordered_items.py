from __future__ import absolute_import
import re
from ruamel import yaml
from googlecloudsdk.third_party.appengine._internal import six_subset
def ordered_items(self):
    result = []
    for key in self.keys:
        if self.get(key) is not None:
            result.append((key, self.get(key)))
    return result