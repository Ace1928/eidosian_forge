from __future__ import absolute_import, division, print_function
import json
import os
import shutil
from ansible.module_utils.six import raise_from
def run_generate_systemd_command(module, module_params, name, version):
    """Generate systemd unit file."""
    command = [module_params['executable'], 'generate', 'systemd', name, '--format', 'json']
    sysconf = module_params['generate_systemd']
    gt4ver = LooseVersion(version) >= LooseVersion('4.0.0')
    if sysconf.get('restart_policy'):
        if sysconf.get('restart_policy') not in ['no', 'on-success', 'on-failure', 'on-abnormal', 'on-watchdog', 'on-abort', 'always']:
            module.fail_json('Restart policy for systemd unit file is "%s" and must be one of: "no", "on-success", "on-failure", "on-abnormal", "on-watchdog", "on-abort", or "always"' % sysconf.get('restart_policy'))
        command.extend(['--restart-policy', sysconf['restart_policy']])
    if sysconf.get('restart_sec') is not None:
        command.extend(['--restart-sec=%s' % sysconf['restart_sec']])
    if sysconf.get('stop_timeout') is not None or sysconf.get('time') is not None:
        arg_name = 'stop-timeout' if gt4ver else 'time'
        arg_value = sysconf.get('stop_timeout') if sysconf.get('stop_timeout') is not None else sysconf.get('time')
        command.extend(['--%s=%s' % (arg_name, arg_value)])
    if sysconf.get('start_timeout') is not None:
        command.extend(['--start-timeout=%s' % sysconf['start_timeout']])
    if sysconf.get('no_header'):
        command.extend(['--no-header'])
    if sysconf.get('names', True):
        command.extend(['--name'])
    if sysconf.get('new'):
        command.extend(['--new'])
    if sysconf.get('container_prefix') is not None:
        command.extend(['--container-prefix=%s' % sysconf['container_prefix']])
    if sysconf.get('pod_prefix') is not None:
        command.extend(['--pod-prefix=%s' % sysconf['pod_prefix']])
    if sysconf.get('separator') is not None:
        command.extend(['--separator=%s' % sysconf['separator']])
    if sysconf.get('after') is not None:
        sys_after = sysconf['after']
        if isinstance(sys_after, str):
            sys_after = [sys_after]
        for after in sys_after:
            command.extend(['--after=%s' % after])
    if sysconf.get('wants') is not None:
        sys_wants = sysconf['wants']
        if isinstance(sys_wants, str):
            sys_wants = [sys_wants]
        for want in sys_wants:
            command.extend(['--wants=%s' % want])
    if sysconf.get('requires') is not None:
        sys_req = sysconf['requires']
        if isinstance(sys_req, str):
            sys_req = [sys_req]
        for require in sys_req:
            command.extend(['--requires=%s' % require])
    for param in ['after', 'wants', 'requires']:
        if sysconf.get(param) is not None and (not gt4ver):
            module.fail_json(msg="Systemd parameter '%s' is supported from podman version 4 only! Current version is %s" % (param, version))
    if module.params['debug'] or module_params['debug']:
        module.log('PODMAN-CONTAINER-DEBUG: systemd command: %s' % ' '.join(command))
    rc, systemd, err = module.run_command(command)
    return (rc, systemd, err)