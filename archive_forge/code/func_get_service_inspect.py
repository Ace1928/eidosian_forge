from __future__ import (absolute_import, division, print_function)
import json
from time import sleep
from ansible.module_utils.common.text.converters import to_native
from ansible_collections.community.docker.plugins.module_utils.version import LooseVersion
from ansible_collections.community.docker.plugins.module_utils.common import AnsibleDockerClient
def get_service_inspect(self, service_id, skip_missing=False):
    """
        Returns Swarm service info as in 'docker service inspect' command about single service

        :param service_id: service ID or name
        :param skip_missing: if True then function will return None instead of failing the task
        :return:
            Single service information structure
        """
    try:
        service_info = self.inspect_service(service_id)
    except NotFound as exc:
        if skip_missing is False:
            self.fail('Error while reading from Swarm manager: %s' % to_native(exc))
        else:
            return None
    except APIError as exc:
        if exc.status_code == 503:
            self.fail('Cannot inspect service: To inspect service execute module on Swarm Manager')
        self.fail('Error inspecting swarm service: %s' % exc)
    except Exception as exc:
        self.fail('Error inspecting swarm service: %s' % exc)
    json_str = json.dumps(service_info, ensure_ascii=False)
    service_info = json.loads(json_str)
    return service_info