from __future__ import absolute_import, division, print_function
import re  # noqa: F402
from ansible.module_utils.basic import AnsibleModule  # noqa: F402
def discover_pods(self):
    pod_name = ''
    if self.module.params['kube_file']:
        if HAS_YAML:
            with open(self.module.params['kube_file']) as f:
                pod = yaml.safe_load(f)
            if 'metadata' in pod:
                pod_name = pod['metadata'].get('name')
            else:
                self.module.fail_json('No metadata in Kube file!\n%s' % pod)
        else:
            with open(self.module.params['kube_file']) as text:
                re_pod_name = re.compile('^\\s{2,4}name: ["|\\\']?(?P<pod_name>[\\w|\\-|\\_]+)["|\\\']?', re.MULTILINE)
                re_pod = re_pod_name.search(text.read())
                if re_pod:
                    pod_name = re_pod.group(1)
    if not pod_name:
        self.module.fail_json("Deployment doesn't have a name!")
    all_pods = ''
    for name in ('name=%s$', 'name=%s-pod-*'):
        cmd = [self.executable, 'pod', 'ps', '-q', '--filter', name % pod_name]
        rc, out, err = self._command_run(cmd)
        all_pods += out
    ids = list(set([i for i in all_pods.splitlines() if i]))
    return ids