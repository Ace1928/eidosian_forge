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
@property
def rest_client(self):
    if not self._rest_client:
        configuration = copy.deepcopy(self.client.configuration)
        validate_certs = self.params.get('registry_validate_certs')
        ssl_ca_cert = self.params.get('registry_ca_cert')
        if validate_certs is not None:
            configuration.verify_ssl = validate_certs
        if ssl_ca_cert is not None:
            configuration.ssl_ca_cert = ssl_ca_cert
        self._rest_client = rest.RESTClientObject(configuration)
    return self._rest_client