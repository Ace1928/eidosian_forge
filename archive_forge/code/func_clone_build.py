from __future__ import (absolute_import, division, print_function)
from datetime import datetime, timezone, timedelta
import traceback
import time
from ansible.module_utils._text import to_native
from ansible_collections.community.okd.plugins.module_utils.openshift_common import AnsibleOpenshiftModule
def clone_build(self, name, namespace, request):
    try:
        result = self.request(method='POST', path='/apis/build.openshift.io/v1/namespaces/{namespace}/builds/{name}/clone'.format(namespace=namespace, name=name), body=request, content_type='application/json')
        return result.to_dict()
    except DynamicApiError as exc:
        msg = 'Failed to clone Build %s/%s due to: %s' % (namespace, name, exc.body)
        self.fail_json(msg=msg, status=exc.status, reason=exc.reason)
    except Exception as e:
        msg = 'Failed to clone Build %s/%s due to: %s' % (namespace, name, to_native(e))
        self.fail_json(msg=msg, error=to_native(e), exception=e)