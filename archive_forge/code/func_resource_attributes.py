from pprint import pformat
from six import iteritems
import re
@resource_attributes.setter
def resource_attributes(self, resource_attributes):
    """
        Sets the resource_attributes of this V1SelfSubjectAccessReviewSpec.
        ResourceAuthorizationAttributes describes information for a resource
        access request

        :param resource_attributes: The resource_attributes of this
        V1SelfSubjectAccessReviewSpec.
        :type: V1ResourceAttributes
        """
    self._resource_attributes = resource_attributes