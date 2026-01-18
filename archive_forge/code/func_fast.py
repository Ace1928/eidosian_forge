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
def fast(image, directory, threads=1, cache_directory=None):
    """Produce a FromDisk compatible file layout under the provided directory.

  After calling this, the following filesystem will exist:
    directory/
      config.json   <-- only *.json, the image's config
      digest        <-- sha256 digest of the image's manifest
      manifest.json <-- the image's manifest
      001.tar.gz    <-- the first layer's .tar.gz filesystem delta
      001.sha256    <-- the sha256 of 1.tar.gz with a "sha256:" prefix.
      ...
      N.tar.gz      <-- the Nth layer's .tar.gz filesystem delta
      N.sha256      <-- the sha256 of N.tar.gz with a "sha256:" prefix.

  We pad layer indices to only 3 digits because of a known ceiling on the number
  of filesystem layers Docker supports.

  Args:
    image: a docker image to save.
    directory: an existing empty directory under which to save the layout.
    threads: the number of threads to use when performing the upload.
    cache_directory: directory that stores file cache.

  Returns:
    A tuple whose first element is the path to the config file, and whose second
    element is an ordered list of tuples whose elements are the filenames
    containing: (.sha256, .tar.gz) respectively.
  """

    def write_file(name, accessor, arg):
        with io.open(name, u'wb') as f:
            f.write(accessor(arg))

    def write_file_and_store(name, accessor, arg, cached_layer):
        write_file(cached_layer, accessor, arg)
        link(cached_layer, name)

    def link(source, dest):
        """Creates a symbolic link dest pointing to source.

    Unlinks first to remove "old" layers if needed
    e.g., image A latest has layers 1, 2 and 3
    after a while it has layers 1, 2 and 3'.
    Since in both cases the layers are named 001, 002 and 003,
    unlinking promises the correct layers are linked in the image directory.

    Args:
      source: image directory source.
      dest: image directory destination.
    """
        try:
            os.symlink(source, dest)
        except OSError as e:
            if e.errno == errno.EEXIST:
                os.unlink(dest)
                os.symlink(source, dest)
            else:
                raise e

    def valid(cached_layer, digest):
        with io.open(cached_layer, u'rb') as f:
            current_digest = docker_digest.SHA256(f.read(), '')
        return current_digest == digest
    with concurrent.futures.ThreadPoolExecutor(max_workers=threads) as executor:
        future_to_params = {}
        config_file = os.path.join(directory, 'config.json')
        f = executor.submit(write_file, config_file, lambda unused: image.config_file().encode('utf8'), 'unused')
        future_to_params[f] = config_file
        executor.submit(write_file, os.path.join(directory, 'digest'), lambda unused: image.digest().encode('utf8'), 'unused')
        executor.submit(write_file, os.path.join(directory, 'manifest.json'), lambda unused: image.manifest().encode('utf8'), 'unused')
        idx = 0
        layers = []
        for blob in reversed(image.fs_layers()):
            layer_name = os.path.join(directory, '%03d.tar.gz' % idx)
            digest_name = os.path.join(directory, '%03d.sha256' % idx)
            digest = blob[7:].encode('utf8')
            f = executor.submit(write_file, digest_name, lambda blob: blob[7:].encode('utf8'), blob)
            future_to_params[f] = digest_name
            digest_str = str(digest)
            if cache_directory:
                cached_layer = os.path.join(cache_directory, digest_str)
                if os.path.exists(cached_layer) and valid(cached_layer, digest_str):
                    f = executor.submit(link, cached_layer, layer_name)
                    future_to_params[f] = layer_name
                else:
                    f = executor.submit(write_file_and_store, layer_name, image.blob, blob, cached_layer)
                    future_to_params[f] = layer_name
            else:
                f = executor.submit(write_file, layer_name, image.blob, blob)
                future_to_params[f] = layer_name
            layers.append((digest_name, layer_name))
            idx += 1
        for future in concurrent.futures.as_completed(future_to_params):
            future.result()
    return (config_file, layers)