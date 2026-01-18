import heapq
import inspect
import unittest
from pbr.version import VersionInfo
def neededResources(self):
    """Return the resources needed for this resource, including self.

        :return: A list of needed resources, in topological deepest-first
            order.
        """
    seen = set([self])
    result = []
    for name, resource in self.resources:
        for resource in resource.neededResources():
            if resource in seen:
                continue
            seen.add(resource)
            result.append(resource)
    result.append(self)
    return result