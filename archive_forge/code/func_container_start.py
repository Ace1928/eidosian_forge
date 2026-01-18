from __future__ import absolute_import, division, print_function
import re
from time import sleep
from ansible.module_utils.common.text.converters import to_native, to_text
from ansible_collections.community.docker.plugins.module_utils.util import (
from ansible_collections.community.docker.plugins.module_utils._api.utils.utils import parse_repository_tag
def container_start(self, container_id):
    self.log('start container %s' % container_id)
    self.results['actions'].append(dict(started=container_id))
    self.results['changed'] = True
    if not self.check_mode:
        try:
            self.engine_driver.start_container(self.client, container_id)
        except Exception as exc:
            self.fail('Error starting container %s: %s' % (container_id, to_native(exc)))
        if self.module.params['detach'] is False:
            status = self.engine_driver.wait_for_container(self.client, container_id)
            self.client.fail_results['status'] = status
            self.results['status'] = status
            if self.module.params['auto_remove']:
                output = 'Cannot retrieve result as auto_remove is enabled'
                if self.param_output_logs:
                    self.module.warn('Cannot output_logs if auto_remove is enabled!')
            else:
                output, real_output = self.engine_driver.get_container_output(self.client, container_id)
                if real_output and self.param_output_logs:
                    self._output_logs(msg=output)
            if self.param_cleanup:
                self.container_remove(container_id, force=True)
            insp = self._get_container(container_id)
            if insp.raw:
                insp.raw['Output'] = output
            else:
                insp.raw = dict(Output=output)
            if status != 0:
                self.results['failed'] = True
                self.results['msg'] = output
            return insp
    return self._get_container(container_id)