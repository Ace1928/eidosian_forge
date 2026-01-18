import datetime
import json
import random
import re
import time
import traceback
import uuid
import typing as t
from ansible.errors import AnsibleConnectionFailure, AnsibleError
from ansible.module_utils.common.text.converters import to_text
from ansible.plugins.connection import ConnectionBase
from ansible.utils.display import Display
def reboot_host(task_action: str, connection: ConnectionBase, boot_time_command: str=_DEFAULT_BOOT_TIME_COMMAND, connect_timeout: int=5, msg: str='Reboot initiated by Ansible', post_reboot_delay: int=0, pre_reboot_delay: int=2, reboot_timeout: int=600, test_command: t.Optional[str]=None, previous_boot_time: t.Optional[str]=None) -> t.Dict[str, t.Any]:
    """Reboot a Windows Host.

    Used by action plugins in ansible.windows to reboot a Windows host. It
    takes in the connection plugin so it can run the commands on the targeted
    host and monitor the reboot process. The return dict will have the
    following keys set:

        changed: Whether a change occurred (reboot was done)
        elapsed: Seconds elapsed between the reboot and it coming back online
        failed: Whether a failure occurred
        unreachable: Whether it failed to connect to the host on the first cmd
        rebooted: Whether the host was rebooted

    When failed=True there may be more keys to give some information around
    the failure like msg, exception. There are other keys that might be
    returned as well but they are dependent on the failure that occurred.

    Verbosity levels used:
        2: Message when each reboot step is completed
        4: Connection plugin operations and their results
        5: Raw commands run and the results of those commands
        Debug: Everything, very verbose

    Args:
        task_action: The name of the action plugin that is running for logging.
        connection: The connection plugin to run the reboot commands on.
        boot_time_command: The command to run when getting the boot timeout.
        connect_timeout: Override the connection timeout of the connection
            plugin when polling the rebooted host.
        msg: The message to display to interactive users when rebooting the
            host.
        post_reboot_delay: Seconds to wait after sending the reboot command
            before checking to see if it has returned.
        pre_reboot_delay: Seconds to wait when sending the reboot command.
        reboot_timeout: Seconds to wait while polling for the host to come
            back online.
        test_command: Command to run when the host is back online and
            determines the machine is ready for management. When not defined
            the default command should wait until the reboot is complete and
            all pre-login configuration has completed.
        previous_boot_time: The previous boot time of the host, when set the
            value is used as the previous boot time check and the code will
            not initiate the reboot itself as it expects the host to be in a
            reboot cycle itself. Used when a module has initiated the reboot.

    Returns:
        (Dict[str, Any]): The return result as a dictionary. Use the 'failed'
            key to determine if there was a failure or not.
    """
    result: t.Dict[str, t.Any] = {'changed': False, 'elapsed': 0, 'failed': False, 'unreachable': False, 'rebooted': False}
    host_context = {'do_close_on_reset': True}
    send_reboot_command = previous_boot_time is None
    if send_reboot_command:
        try:
            previous_boot_time = _do_until_success_or_retry_limit(task_action, connection, host_context, 'pre-reboot boot time check', 3, _get_system_boot_time, task_action, connection, boot_time_command)
        except Exception as e:
            if isinstance(e, _ReturnResultException):
                result.update(e.result)
            if isinstance(e, AnsibleConnectionFailure):
                result['unreachable'] = True
            else:
                result['failed'] = True
            result['msg'] = str(e)
            result['exception'] = traceback.format_exc()
            return result
    original_connection_timeout: t.Optional[float] = None
    try:
        original_connection_timeout = connection.get_option('connection_timeout')
        display.vvvv(f'{task_action}: saving original connection_timeout of {original_connection_timeout}')
    except KeyError:
        display.vvvv(f'{task_action}: connection_timeout connection option has not been set')
    reboot_command = '$ErrorActionPreference = \'Continue\'\n\nif ($%s) {\n    Remove-Item -LiteralPath \'%s\' -Force -ErrorAction SilentlyContinue\n}\n\n$stdout = $null\n$stderr = . { shutdown.exe /r /t %s /c %s | Set-Variable stdout } 2>&1 | ForEach-Object ToString\n\nConvertTo-Json -Compress -InputObject @{\n    stdout = (@($stdout) -join "`n")\n    stderr = (@($stderr) -join "`n")\n    rc = $LASTEXITCODE\n}\n' % (str(not test_command), _LOGON_UI_KEY, int(pre_reboot_delay), _quote_pwsh(msg))
    expected_test_result = None
    if not test_command:
        expected_test_result = f'success-{uuid.uuid4()}'
        test_command = f"Get-Item -LiteralPath '{_LOGON_UI_KEY}' -ErrorAction Stop; '{expected_test_result}'"
    start = None
    try:
        if send_reboot_command:
            _perform_reboot(task_action, connection, reboot_command)
        start = datetime.datetime.utcnow()
        result['changed'] = True
        result['rebooted'] = True
        if post_reboot_delay != 0:
            display.vv(f'{task_action}: waiting an additional {post_reboot_delay} seconds')
            time.sleep(post_reboot_delay)
        display.vv(f'{task_action} validating reboot')
        _do_until_success_or_timeout(task_action, connection, host_context, 'last boot time check', reboot_timeout, _check_boot_time, task_action, connection, host_context, previous_boot_time, boot_time_command, connect_timeout)
        if original_connection_timeout is not None:
            _set_connection_timeout(task_action, connection, host_context, original_connection_timeout)
        display.vv(f'{task_action} running post reboot test command')
        _do_until_success_or_timeout(task_action, connection, host_context, 'post-reboot test command', reboot_timeout, _run_test_command, task_action, connection, test_command, expected=expected_test_result)
        display.vv(f'{task_action}: system successfully rebooted')
    except Exception as e:
        if isinstance(e, _ReturnResultException):
            result.update(e.result)
        result['failed'] = True
        result['msg'] = str(e)
        result['exception'] = traceback.format_exc()
    if start:
        elapsed = datetime.datetime.utcnow() - start
        result['elapsed'] = elapsed.seconds
    return result