from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import collections
import json
from typing import Mapping, Sequence
from googlecloudsdk.api_lib.run import k8s_object
@dependencies.setter
def dependencies(self, dependencies: Mapping[str, Sequence[str]]):
    """Sets the resource's container dependencies.

    Args:
      dependencies: A dictionary mapping containers to a list of their
        dependencies by name.

    Container dependencies are stored in the
    'run.googleapis.com/container-dependencies' annotation as json. Setting an
    empty set of dependencies will clear this annotation.
    """
    if dependencies:
        self.annotations[k8s_object.CONTAINER_DEPENDENCIES_ANNOTATION] = json.dumps({k: list(v) for k, v in dependencies.items()})
    elif k8s_object.CONTAINER_DEPENDENCIES_ANNOTATION in self.annotations:
        del self.annotations[k8s_object.CONTAINER_DEPENDENCIES_ANNOTATION]