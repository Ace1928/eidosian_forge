from __future__ import absolute_import, division, print_function
import re
from time import sleep
from ansible.module_utils.common.text.converters import to_native, to_text
from ansible_collections.community.docker.plugins.module_utils.util import (
from ansible_collections.community.docker.plugins.module_utils._api.utils.utils import parse_repository_tag
def has_different_configuration(self, container, container_image, image, host_info):
    differences = DifferenceTracker()
    update_differences = DifferenceTracker()
    for options, param_values in self.parameters:
        engine = options.get_engine(self.engine_driver.name)
        if engine.can_update_value(self.engine_driver.get_api_version(self.client)):
            self._record_differences(update_differences, options, param_values, engine, container, container_image, image, host_info)
        else:
            self._record_differences(differences, options, param_values, engine, container, container_image, image, host_info)
    has_differences = not differences.empty
    if has_differences:
        differences.merge(update_differences)
    return (has_differences, differences)