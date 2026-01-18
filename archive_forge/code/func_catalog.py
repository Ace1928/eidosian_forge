from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import abc
import gzip
import io
import json
import os
import tarfile
import threading
from containerregistry.client import docker_creds
from containerregistry.client import docker_name
from containerregistry.client.v2_2 import docker_digest
from containerregistry.client.v2_2 import docker_http
import httplib2
import six
from six.moves import zip  # pylint: disable=redefined-builtin
import six.moves.http_client
def catalog(self, page_size=100):
    if isinstance(self._name, docker_name.Repository):
        raise ValueError('Expected docker_name.Registry for "name"')
    url = '{scheme}://{registry}/v2/_catalog?n={page_size}'.format(scheme=docker_http.Scheme(self._name.registry), registry=self._name.registry, page_size=page_size)
    for _, content in self._transport.PaginatedRequest(url, accepted_codes=[six.moves.http_client.OK]):
        wrapper_object = json.loads(content.decode('utf8'))
        if 'repositories' not in wrapper_object:
            raise docker_http.BadStateException('Malformed JSON response: %s' % content)
        for repo in wrapper_object['repositories']:
            yield repo