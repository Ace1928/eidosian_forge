from __future__ import (absolute_import, division, print_function)
from ansible_collections.fortinet.fortios.plugins.module_utils.fortimanager.common import FMGR_RC
from ansible_collections.fortinet.fortios.plugins.module_utils.fortimanager.common import FMGBaseException
from ansible_collections.fortinet.fortios.plugins.module_utils.fortimanager.common import FMGRCommon
from ansible_collections.fortinet.fortios.plugins.module_utils.fortimanager.common import scrub_dict
@staticmethod
def return_response(module, results, msg='NULL', good_codes=(0,), stop_on_fail=True, stop_on_success=False, skipped=False, changed=False, unreachable=False, failed=False, success=False, changed_if_success=True, ansible_facts=()):
    """
        This function controls the logout and error reporting after an method or function runs. The exit_json for
        ansible comes from logic within this function. If this function returns just the msg, it means to continue
        execution on the playbook. It is called from the ansible module, or from the self.govern_response function.

        :param module: The Ansible Module CLASS object, used to run fail/exit json
        :type module: object
        :param msg: An overridable custom message from the module that called this.
        :type msg: string
        :param results: A dictionary object containing an API call results
        :type results: dict
        :param good_codes: A list of exit codes considered successful from FortiManager
        :type good_codes: list
        :param stop_on_fail: If true, stops playbook run when return code is NOT IN good codes (default: true)
        :type stop_on_fail: boolean
        :param stop_on_success: If true, stops playbook run when return code is IN good codes (default: false)
        :type stop_on_success: boolean
        :param changed: If True, tells Ansible that object was changed (default: false)
        :type skipped: boolean
        :param skipped: If True, tells Ansible that object was skipped (default: false)
        :type skipped: boolean
        :param unreachable: If True, tells Ansible that object was unreachable (default: false)
        :type unreachable: boolean
        :param failed: If True, tells Ansible that execution was a failure. Overrides good_codes. (default: false)
        :type unreachable: boolean
        :param success: If True, tells Ansible that execution was a success. Overrides good_codes. (default: false)
        :type unreachable: boolean
        :param changed_if_success: If True, defaults to changed if successful if you specify or not"
        :type changed_if_success: boolean
        :param ansible_facts: A prepared dictionary of ansible facts from the execution.
        :type ansible_facts: dict

        :return: A string object that contains an error message
        :rtype: str
        """
    if len(results) == 0 or (failed and success) or (changed and unreachable):
        module.exit_json(msg='Handle_response was called with no results, or conflicting failed/success or changed/unreachable parameters. Fix the exit code on module. Generic Failure', failed=True)
    if not failed and (not success):
        if len(results) > 0:
            if results[0] not in good_codes:
                failed = True
            elif results[0] in good_codes:
                success = True
    if len(results) > 0:
        if msg == 'NULL':
            try:
                msg = results[1]['status']['message']
            except BaseException:
                msg = 'No status message returned at results[1][status][message], and none supplied to msg parameter for handle_response.'
        if failed:
            if failed and skipped:
                failed = False
            if failed and unreachable:
                failed = False
            if stop_on_fail:
                module.exit_json(msg=msg, failed=failed, changed=changed, unreachable=unreachable, skipped=skipped, results=results[1], ansible_facts=ansible_facts, rc=results[0], invocation={'module_args': ansible_facts['ansible_params']})
        elif success:
            if changed_if_success:
                changed = True
                success = False
            if stop_on_success:
                module.exit_json(msg=msg, success=success, changed=changed, unreachable=unreachable, skipped=skipped, results=results[1], ansible_facts=ansible_facts, rc=results[0], invocation={'module_args': ansible_facts['ansible_params']})
    return msg