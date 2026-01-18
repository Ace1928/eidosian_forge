from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import abc
import json
from containerregistry.client import docker_creds
from containerregistry.client import docker_name
from containerregistry.client.v2_2 import docker_digest
from containerregistry.client.v2_2 import docker_http
from containerregistry.client.v2_2 import docker_image as v2_2_image
import httplib2
import six
import six.moves.http_client
def images(self):
    """Returns a list of tuples whose elements are (name, platform, image).

    Raises:
      InvalidMediaTypeError: a child with an unexpected media type was found.
    """
    manifests = json.loads(self.manifest())['manifests']
    results = []
    for entry in manifests:
        digest = entry['digest']
        base = self._name.as_repository()
        name = docker_name.Digest('{base}@{digest}'.format(base=base, digest=digest))
        media_type = entry['mediaType']
        if media_type in docker_http.MANIFEST_LIST_MIMES:
            image = FromRegistry(name, self._creds, self._original_transport)
        elif media_type in docker_http.SUPPORTED_MANIFEST_MIMES:
            image = v2_2_image.FromRegistry(name, self._creds, self._original_transport, [media_type])
        else:
            raise InvalidMediaTypeError('Invalid media type: ' + media_type)
        platform = Platform(entry['platform']) if 'platform' in entry else None
        results.append((name, platform, image))
    return results