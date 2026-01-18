from __future__ import absolute_import, division, print_function
import json
import re
import shlex
from ansible.module_utils._text import to_native
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.containers.podman.plugins.module_utils.podman.common import run_podman_command
def pull_image(self, image_name=None):
    if image_name is None:
        image_name = self.image_name
    args = ['pull', image_name, '-q']
    if self.arch:
        args.extend(['--arch', self.arch])
    if self.auth_file:
        args.extend(['--authfile', self.auth_file])
    if self.username and self.password:
        cred_string = '{user}:{password}'.format(user=self.username, password=self.password)
        args.extend(['--creds', cred_string])
    if self.validate_certs is not None:
        if self.validate_certs:
            args.append('--tls-verify')
        else:
            args.append('--tls-verify=false')
    if self.ca_cert_dir:
        args.extend(['--cert-dir', self.ca_cert_dir])
    rc, out, err = self._run(args, ignore_errors=True)
    if rc != 0:
        if not self.pull:
            self.module.fail_json(msg='Failed to find image {image_name} locally, image pull set to {pull_bool}'.format(pull_bool=self.pull, image_name=image_name))
        else:
            self.module.fail_json(msg='Failed to pull image {image_name}'.format(image_name=image_name))
    return self.inspect_image(out.strip())