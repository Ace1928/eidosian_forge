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
def check_usage_only(self):
    response = json.loads(self._content('tags/list?check_usage_only=true').decode('utf8'))
    if 'usage' not in response:
        raise docker_http.BadStateException('Malformed JSON response: {}. Missing "usage" field'.format(response))
    return response.get('usage')