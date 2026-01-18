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
def compatible_with(self, target):
    """Returns True if this platform can run on the 'target' platform."""
    return target.can_run(self)