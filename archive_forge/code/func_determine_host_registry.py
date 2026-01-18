from __future__ import (absolute_import, division, print_function)
from datetime import datetime, timezone, timedelta
import traceback
import copy
from ansible.module_utils._text import to_native
from ansible.module_utils.parsing.convert_bool import boolean
from ansible.module_utils.six import iteritems
from ansible_collections.community.okd.plugins.module_utils.openshift_common import AnsibleOpenshiftModule
from ansible_collections.community.okd.plugins.module_utils.openshift_images_common import (
from ansible_collections.community.okd.plugins.module_utils.openshift_docker_image import (
def determine_host_registry(module, images, image_streams):

    def _f_managed_images(obj):
        value = read_object_annotation(obj, 'openshift.io/image.managed')
        return boolean(value) if value is not None else False
    managed_images = list(filter(_f_managed_images, images))
    sorted_images = sorted(managed_images, key=lambda x: x['metadata']['creationTimestamp'], reverse=True)
    docker_image_ref = ''
    if len(sorted_images) > 0:
        docker_image_ref = sorted_images[0].get('dockerImageReference', '')
    else:
        sorted_image_streams = sorted(image_streams, key=lambda x: x['metadata']['creationTimestamp'], reverse=True)
        for i_stream in sorted_image_streams:
            docker_image_ref = i_stream['status'].get('dockerImageRepository', '')
            if len(docker_image_ref) > 0:
                break
    if len(docker_image_ref) == 0:
        module.exit_json(changed=False, result='no managed image found')
    result, error = parse_docker_image_ref(docker_image_ref, module)
    return result['hostname']