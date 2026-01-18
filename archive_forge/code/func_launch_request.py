import os
import sys
import debugpy
from debugpy import launcher
from debugpy.common import json
from debugpy.launcher import debuggee
def launch_request(request):
    debug_options = set(request('debugOptions', json.array(str)))

    def property_or_debug_option(prop_name, flag_name):
        assert prop_name[0].islower() and flag_name[0].isupper()
        value = request(prop_name, bool, optional=True)
        if value == ():
            value = None
        if flag_name in debug_options:
            if value is False:
                raise request.isnt_valid('{0}:false and "debugOptions":[{1}] are mutually exclusive', json.repr(prop_name), json.repr(flag_name))
            value = True
        return value
    python = request('python', json.array(str, size=(1,)))
    cmdline = list(python)
    if not request('noDebug', json.default(False)):
        if sys.version_info[:2] >= (3, 11):
            cmdline += ['-X', 'frozen_modules=off']
        port = request('port', int)
        cmdline += [os.path.dirname(debugpy.__file__), '--connect', launcher.adapter_host + ':' + str(port)]
        if not request('subProcess', True):
            cmdline += ['--configure-subProcess', 'False']
        qt_mode = request('qt', json.enum('none', 'auto', 'pyside', 'pyside2', 'pyqt4', 'pyqt5', optional=True))
        cmdline += ['--configure-qt', qt_mode]
        adapter_access_token = request('adapterAccessToken', str, optional=True)
        if adapter_access_token != ():
            cmdline += ['--adapter-access-token', adapter_access_token]
        debugpy_args = request('debugpyArgs', json.array(str))
        cmdline += debugpy_args
    cmdline += sys.argv[1:]
    process_name = request('processName', sys.executable)
    env = os.environ.copy()
    env_changes = request('env', json.object((str, type(None))))
    if sys.platform == 'win32':
        env = {k.upper(): v for k, v in os.environ.items()}
        new_env_changes = {}
        for k, v in env_changes.items():
            k_upper = k.upper()
            if k_upper in new_env_changes:
                if new_env_changes[k_upper] == v:
                    continue
                else:
                    raise request.isnt_valid('Found duplicate in "env": {0}.'.format(k_upper))
            new_env_changes[k_upper] = v
        env_changes = new_env_changes
    if 'DEBUGPY_TEST' in env:
        env.pop('COV_CORE_SOURCE', None)
    env.update(env_changes)
    env = {k: v for k, v in env.items() if v is not None}
    if request('gevent', False):
        env['GEVENT_SUPPORT'] = 'True'
    console = request('console', json.enum('internalConsole', 'integratedTerminal', 'externalTerminal', optional=True))
    redirect_output = property_or_debug_option('redirectOutput', 'RedirectOutput')
    if redirect_output is None:
        redirect_output = console == 'internalConsole'
    if redirect_output:
        env['PYTHONUNBUFFERED'] = '1'
        env['PYTHONIOENCODING'] = 'utf-8'
    if property_or_debug_option('waitOnNormalExit', 'WaitOnNormalExit'):
        if console == 'internalConsole':
            raise request.isnt_valid('"waitOnNormalExit" is not supported for "console":"internalConsole"')
        debuggee.wait_on_exit_predicates.append(lambda code: code == 0)
    if property_or_debug_option('waitOnAbnormalExit', 'WaitOnAbnormalExit'):
        if console == 'internalConsole':
            raise request.isnt_valid('"waitOnAbnormalExit" is not supported for "console":"internalConsole"')
        debuggee.wait_on_exit_predicates.append(lambda code: code != 0)
    debuggee.spawn(process_name, cmdline, env, redirect_output)
    return {}