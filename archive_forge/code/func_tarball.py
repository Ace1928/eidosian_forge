from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import errno
import io
import json
import os
import tarfile
import concurrent.futures
from containerregistry.client import docker_name
from containerregistry.client.v1 import docker_image as v1_image
from containerregistry.client.v1 import save as v1_save
from containerregistry.client.v2 import v1_compat
from containerregistry.client.v2_2 import docker_digest
from containerregistry.client.v2_2 import docker_http
from containerregistry.client.v2_2 import docker_image as v2_2_image
from containerregistry.client.v2_2 import v2_compat
import six
def tarball(name, image, tar):
    """Produce a "docker save" compatible tarball from the DockerImage.

  Args:
    name: The tag name to write into repositories and manifest.json
    image: a docker image to save.
    tar: the open tarfile into which we are writing the image tarball.
  """
    multi_image_tarball({name: image}, tar, {})