from __future__ import (absolute_import, division, print_function)
from datetime import datetime, timezone, timedelta
import traceback
import time
from ansible.module_utils._text import to_native
from ansible_collections.community.okd.plugins.module_utils.openshift_common import AnsibleOpenshiftModule
def cancel_build(self, restart):
    kind = 'Build'
    api_version = 'build.openshift.io/v1'
    namespace = self.params.get('namespace')
    phases = ['new', 'pending', 'running']
    build_phases = self.params.get('build_phases', [])
    if build_phases:
        phases = [p.lower() for p in build_phases]
    names = []
    if self.params.get('build_name'):
        names.append(self.params.get('build_name'))
    else:
        build_config = self.params.get('build_config_name')
        params = dict(kind=kind, api_version=api_version, namespace=namespace)
        resources = self.kubernetes_facts(**params).get('resources', [])

        def _filter_builds(build):
            config = build['metadata'].get('labels', {}).get('openshift.io/build-config.name')
            return build_config is None or (build_config is not None and config in build_config)
        for item in list(filter(_filter_builds, resources)):
            name = item['metadata']['name']
            if name not in names:
                names.append(name)
    if len(names) == 0:
        self.exit_json(changed=False, msg='No Build found from namespace %s' % namespace)
    warning = []
    builds_to_cancel = []
    for name in names:
        params = dict(kind=kind, api_version=api_version, name=name, namespace=namespace)
        resource = self.kubernetes_facts(**params).get('resources', [])
        if len(resource) == 0:
            warning.append('Build %s/%s not found' % (namespace, name))
            continue
        resource = resource[0]
        phase = resource['status'].get('phase').lower()
        if phase in phases:
            builds_to_cancel.append(resource)
        else:
            warning.append('build %s/%s is not in expected phase, found %s' % (namespace, name, phase))
    changed = False
    result = []
    for build in builds_to_cancel:
        build['status']['cancelled'] = True
        name = build['metadata']['name']
        changed = True
        try:
            content_type = 'application/json'
            cancelled_build = self.request('PUT', '/apis/build.openshift.io/v1/namespaces/{0}/builds/{1}'.format(namespace, name), body=build, content_type=content_type).to_dict()
            result.append(cancelled_build)
        except DynamicApiError as exc:
            self.fail_json(msg='Failed to cancel Build %s/%s due to: %s' % (namespace, name, exc), reason=exc.reason, status=exc.status)
        except Exception as e:
            self.fail_json(msg='Failed to cancel Build %s/%s due to: %s' % (namespace, name, e))

    def _wait_until_cancelled(build, wait_timeout, wait_sleep):
        start = datetime.now()
        last_phase = None
        name = build['metadata']['name']
        while (datetime.now() - start).seconds < wait_timeout:
            params = dict(kind=kind, api_version=api_version, name=name, namespace=namespace)
            resource = self.kubernetes_facts(**params).get('resources', [])
            if len(resource) == 0:
                return (None, 'Build %s/%s not found' % (namespace, name))
            resource = resource[0]
            last_phase = resource['status']['phase']
            if last_phase == 'Cancelled':
                return (resource, None)
            time.sleep(wait_sleep)
        return (None, 'Build %s/%s is not cancelled as expected, current state is %s' % (namespace, name, last_phase))
    if result and self.params.get('wait'):
        wait_timeout = self.params.get('wait_timeout')
        wait_sleep = self.params.get('wait_sleep')
        wait_result = []
        for build in result:
            ret, err = _wait_until_cancelled(build, wait_timeout, wait_sleep)
            if err:
                self.exit_json(msg=err)
            wait_result.append(ret)
        result = wait_result
    if restart:
        self.start_build()
    self.exit_json(builds=result, changed=changed)