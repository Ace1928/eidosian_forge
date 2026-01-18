from __future__ import (absolute_import, division, print_function)
from datetime import datetime
from ansible_collections.community.okd.plugins.module_utils.openshift_docker_image import (
from ansible.module_utils.six import iteritems
def get_image_blobs(image):
    blobs = [layer['image'] for layer in image['dockerImageLayers'] if 'image' in layer]
    docker_image_metadata = image.get('dockerImageMetadata')
    if not docker_image_metadata:
        return (blobs, 'failed to read metadata for image %s' % image['metadata']['name'])
    media_type_manifest = ('application/vnd.docker.distribution.manifest.v2+json', 'application/vnd.oci.image.manifest.v1+json')
    media_type_has_config = image['dockerImageManifestMediaType'] in media_type_manifest
    docker_image_id = docker_image_metadata.get('Id')
    if media_type_has_config and docker_image_id and (len(docker_image_id) > 0):
        blobs.append(docker_image_id)
    return (blobs, None)