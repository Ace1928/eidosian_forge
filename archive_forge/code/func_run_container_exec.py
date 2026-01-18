from __future__ import absolute_import, division, print_function
import shlex
from ansible.module_utils.six import string_types
from ansible.module_utils._text import to_text
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.containers.podman.plugins.module_utils.podman.common import run_podman_command
def run_container_exec(module: AnsibleModule) -> dict:
    """
    Execute podman-container-exec for the given options
    """
    exec_with_args = ['container', 'exec']
    changed = True
    exec_options = []
    name = module.params['name']
    argv = module.params['argv']
    command = module.params['command']
    detach = module.params['detach']
    env = module.params['env']
    privileged = module.params['privileged']
    tty = module.params['tty']
    user = module.params['user']
    workdir = module.params['workdir']
    if command is not None:
        argv = shlex.split(command)
    if detach:
        exec_options.append('--detach')
    if env is not None:
        for key, value in env.items():
            if not isinstance(value, string_types):
                module.fail_json(msg='Specify string value %s on the env field' % value)
            to_text(value, errors='surrogate_or_strict')
            exec_options += ['--env', '%s="%s"' % (key, value)]
    if privileged:
        exec_options.append('--privileged')
    if tty:
        exec_options.append('--tty')
    if user is not None:
        exec_options += ['--user', to_text(user, errors='surrogate_or_strict')]
    if workdir is not None:
        exec_options += ['--workdir', to_text(workdir, errors='surrogate_or_strict')]
    exec_options.append(name)
    exec_options.extend(argv)
    exec_with_args.extend(exec_options)
    rc, stdout, stderr = run_podman_command(module=module, executable='podman', args=exec_with_args)
    result = {'changed': changed, 'podman_command': exec_options, 'rc': rc, 'stdout': stdout, 'stderr': stderr}
    if detach:
        result['exec_id'] = stdout.replace('\n', '')
    return result