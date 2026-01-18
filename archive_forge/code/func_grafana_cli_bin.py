from __future__ import absolute_import, division, print_function
import os
from ansible.module_utils.basic import AnsibleModule
def grafana_cli_bin(params):
    """
    Get the grafana-cli binary path with global options.
    Raise a GrafanaCliException if the grafana-cli is not present or not in PATH

    :param params: ansible module params. Used to fill grafana-cli global params.
    """
    program = 'grafana-cli'
    grafana_cli = None

    def is_exe(fpath):
        return os.path.isfile(fpath) and os.access(fpath, os.X_OK)
    fpath, fname = os.path.split(program)
    if fpath:
        if is_exe(program):
            grafana_cli = program
    else:
        for path in os.environ['PATH'].split(os.pathsep):
            path = path.strip('"')
            exe_file = os.path.join(path, program)
            if is_exe(exe_file):
                grafana_cli = exe_file
                break
    if grafana_cli is None:
        raise GrafanaCliException('grafana-cli binary is not present or not in PATH')
    else:
        if 'grafana_plugin_url' in params and params['grafana_plugin_url']:
            grafana_cli = '{0} {1} {2}'.format(grafana_cli, '--pluginUrl', params['grafana_plugin_url'])
        if 'grafana_plugins_dir' in params and params['grafana_plugins_dir']:
            grafana_cli = '{0} {1} {2}'.format(grafana_cli, '--pluginsDir', params['grafana_plugins_dir'])
        if 'grafana_repo' in params and params['grafana_repo']:
            grafana_cli = '{0} {1} {2}'.format(grafana_cli, '--repo', params['grafana_repo'])
        if 'validate_certs' in params and params['validate_certs'] is False:
            grafana_cli = '{0} {1}'.format(grafana_cli, '--insecure')
        return '{0} {1}'.format(grafana_cli, 'plugins')