from __future__ import absolute_import
from ruamel import yaml
from googlecloudsdk.third_party.appengine.api import validation
from googlecloudsdk.third_party.appengine.api import yaml_builder
from googlecloudsdk.third_party.appengine.api import yaml_errors
from googlecloudsdk.third_party.appengine.api import yaml_listener
def see(self, key):
    if key in self.seen:
        raise yaml_errors.DuplicateAttribute("Duplicate attribute '%s'." % key)
    self.seen.add(key)