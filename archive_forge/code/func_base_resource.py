from __future__ import absolute_import, division, print_function
import kubernetes.dynamic
def base_resource(self):
    if self.__base_resource:
        return self.__base_resource
    elif self.base_resource_lookup:
        self.__base_resource = self.client.resources.get(**self.base_resource_lookup)
        return self.__base_resource
    return None