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
def multi_image_tarball(tag_to_image, tar, tag_to_v1_image=None):
    """Produce a "docker save" compatible tarball from the DockerImages.

  Args:
    tag_to_image: A dictionary of tags to the images they label.
    tar: the open tarfile into which we are writing the image tarball.
    tag_to_v1_image: A dictionary of tags to the v1 form of the images
        they label.  If this isn't provided, the image is simply converted.
  """

    def add_file(filename, contents):
        contents_bytes = contents.encode('utf8')
        info = tarfile.TarInfo(filename)
        info.size = len(contents_bytes)
        tar.addfile(tarinfo=info, fileobj=io.BytesIO(contents_bytes))
    tag_to_v1_image = tag_to_v1_image or {}
    manifests = []
    for tag, image in six.iteritems(tag_to_image):
        digest = docker_digest.SHA256(image.config_file().encode('utf8'), '')
        add_file(digest + '.json', image.config_file())
        cfg = json.loads(image.config_file())
        diffs = set(cfg.get('rootfs', {}).get('diff_ids', []))
        v1_img = tag_to_v1_image.get(tag)
        if not v1_img:
            v2_img = v2_compat.V2FromV22(image)
            v1_img = v1_compat.V1FromV2(v2_img)
            tag_to_v1_image[tag] = v1_img
        manifest = {'Config': digest + '.json', 'Layers': [layer_id + '/layer.tar' for layer_id in reversed(v1_img.ancestry(v1_img.top())) if _diff_id(v1_img, layer_id) in diffs and (not json.loads(v1_img.json(layer_id)).get('throwaway'))], 'RepoTags': [str(tag)]}
        layer_sources = {}
        input_manifest = json.loads(image.manifest())
        input_layers = input_manifest['layers']
        for input_layer in input_layers:
            if input_layer['mediaType'] == docker_http.FOREIGN_LAYER_MIME:
                diff_id = image.digest_to_diff_id(input_layer['digest'])
                layer_sources[diff_id] = input_layer
        if layer_sources:
            manifest['LayerSources'] = layer_sources
        manifests.append(manifest)
    v1_save.multi_image_tarball(tag_to_v1_image, tar)
    add_file('manifest.json', json.dumps(manifests, sort_keys=True))