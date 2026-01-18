from __future__ import (absolute_import, division, print_function)
import getpass
import socket
import time
import uuid
from collections import OrderedDict
from contextlib import closing
from os.path import basename
from ansible.errors import AnsibleError, AnsibleRuntimeError
from ansible.module_utils.six import raise_from
from ansible.plugins.callback import CallbackBase
def create_span_data(self, apm_cli, task_data, host_data):
    """ create the span with the given TaskData and HostData """
    name = '[%s] %s: %s' % (host_data.name, task_data.play, task_data.name)
    message = 'success'
    status = 'success'
    enriched_error_message = None
    if host_data.status == 'included':
        rc = 0
    else:
        res = host_data.result._result
        rc = res.get('rc', 0)
        if host_data.status == 'failed':
            message = self.get_error_message(res)
            enriched_error_message = self.enrich_error_message(res)
            status = 'failure'
        elif host_data.status == 'skipped':
            if 'skip_reason' in res:
                message = res['skip_reason']
            else:
                message = 'skipped'
            status = 'unknown'
    with capture_span(task_data.name, start=task_data.start, span_type='ansible.task.run', duration=host_data.finish - task_data.start, labels={'ansible.task.args': task_data.args, 'ansible.task.message': message, 'ansible.task.module': task_data.action, 'ansible.task.name': name, 'ansible.task.result': rc, 'ansible.task.host.name': host_data.name, 'ansible.task.host.status': host_data.status}) as span:
        span.outcome = status
        if 'failure' in status:
            exception = AnsibleRuntimeError(message='{0}: {1} failed with error message {2}'.format(task_data.action, name, enriched_error_message))
            apm_cli.capture_exception(exc_info=(type(exception), exception, exception.__traceback__), handled=True)