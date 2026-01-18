from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import collections
import json
from typing import Mapping, Sequence
from googlecloudsdk.api_lib.run import k8s_object
@property
def containers(self):
    """The containers in the revisionTemplate."""
    return ContainersAsDictionaryWrapper(self.spec.containers, self.volumes, self._messages)