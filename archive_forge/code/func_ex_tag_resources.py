import re
import copy
import time
import base64
import hashlib
from libcloud.utils.py3 import b, httplib
from libcloud.utils.misc import dict2str, str2list, str2dicts, get_secure_random_string
from libcloud.common.base import Response, JsonResponse, ConnectionUserAndKey
from libcloud.common.types import ProviderError, InvalidCredsError
from libcloud.compute.base import Node, KeyPair, NodeSize, NodeImage, NodeDriver, is_private_subnet
from libcloud.compute.types import Provider, NodeState
from libcloud.utils.iso8601 import parse_date
from libcloud.common.cloudsigma import (
def ex_tag_resources(self, resources, tag):
    """
        Associate tag with the provided resources.

        :param resources: Resources to associate a tag with.
        :type resources: ``list`` of :class:`libcloud.compute.base.Node` or
                        :class:`.CloudSigmaDrive`

        :param tag: Tag to associate with the resources.
        :type tag: :class:`.CloudSigmaTag`

        :return: Updated tag object.
        :rtype: :class:`.CloudSigmaTag`
        """
    resources = tag.resources[:]
    for resource in resources:
        if not hasattr(resource, 'id'):
            raise ValueError("Resource doesn't have id attribute")
        resources.append(resource.id)
    resources = list(set(resources))
    data = {'name': tag.name, 'resources': resources}
    action = '/tags/%s/' % tag.id
    response = self.connection.request(action=action, method='PUT', data=data).object
    tag = self._to_tag(data=response)
    return tag