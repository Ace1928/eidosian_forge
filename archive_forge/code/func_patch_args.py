import os
import re
import sys
from _pydev_bundle._pydev_saved_modules import threading
from _pydevd_bundle.pydevd_constants import get_global_debugger, IS_WINDOWS, IS_JYTHON, get_current_thread_id, \
from _pydev_bundle import pydev_log
from contextlib import contextmanager
from _pydevd_bundle import pydevd_constants, pydevd_defaults
from _pydevd_bundle.pydevd_defaults import PydevdCustomization
import ast
def patch_args(args, is_exec=False):
    """
    :param list args:
        Arguments to patch.

    :param bool is_exec:
        If it's an exec, the current process will be replaced (this means we have
        to keep the same ppid).
    """
    try:
        pydev_log.debug('Patching args: %s', args)
        original_args = args
        try:
            unquoted_args = remove_quotes_from_args(args)
        except InvalidTypeInArgsException as e:
            pydev_log.info('Unable to monkey-patch subprocess arguments because a type found in the args is invalid: %s', e)
            return original_args
        del args
        from pydevd import SetupHolder
        if not unquoted_args:
            return original_args
        if not is_python(unquoted_args[0]):
            pydev_log.debug('Process is not python, returning.')
            return original_args
        args_as_str = _get_str_type_compatible('', unquoted_args)
        params_with_value_in_separate_arg = ('--check-hash-based-pycs', '--jit')
        params_with_combinable_arg = set(('W', 'X', 'Q', 'c', 'm'))
        module_name = None
        before_module_flag = ''
        module_name_i_start = -1
        module_name_i_end = -1
        code = None
        code_i = -1
        code_i_end = -1
        code_flag = ''
        filename = None
        filename_i = -1
        ignore_next = True
        for i, arg_as_str in enumerate(args_as_str):
            if ignore_next:
                ignore_next = False
                continue
            if arg_as_str.startswith('-'):
                if arg_as_str == '-':
                    pydev_log.debug('Unable to fix arguments to attach debugger on subprocess when reading from stdin ("python ... -").')
                    return original_args
                if arg_as_str.startswith(params_with_value_in_separate_arg):
                    if arg_as_str in params_with_value_in_separate_arg:
                        ignore_next = True
                    continue
                break_out = False
                for j, c in enumerate(arg_as_str):
                    if c in params_with_combinable_arg:
                        remainder = arg_as_str[j + 1:]
                        if not remainder:
                            ignore_next = True
                        if c == 'm':
                            before_module_flag = arg_as_str[:j]
                            if before_module_flag == '-':
                                before_module_flag = ''
                            module_name_i_start = i
                            if not remainder:
                                module_name = unquoted_args[i + 1]
                                module_name_i_end = i + 1
                            else:
                                module_name = unquoted_args[i][j + 1:]
                                module_name_i_end = module_name_i_start
                            break_out = True
                            break
                        elif c == 'c':
                            code_flag = arg_as_str[:j + 1]
                            if not remainder:
                                code = unquoted_args[i + 1]
                                code_i_end = i + 2
                            else:
                                code = remainder
                                code_i_end = i + 1
                            code_i = i
                            break_out = True
                            break
                        else:
                            break
                if break_out:
                    break
            else:
                filename = unquoted_args[i]
                filename_i = i
                if _is_managed_arg(filename):
                    pydev_log.debug('Skipped monkey-patching as pydevd.py is in args already.')
                    return original_args
                break
        else:
            pydev_log.debug('Unable to fix arguments to attach debugger on subprocess (filename not found).')
            return original_args
        if code_i != -1:
            host, port = _get_host_port()
            if port is not None:
                new_args = []
                new_args.extend(unquoted_args[:code_i])
                new_args.append(code_flag)
                new_args.append(_get_python_c_args(host, port, code, unquoted_args, SetupHolder.setup))
                new_args.extend(unquoted_args[code_i_end:])
                return quote_args(new_args)
        first_non_vm_index = max(filename_i, module_name_i_start)
        if first_non_vm_index == -1:
            pydev_log.debug('Unable to fix arguments to attach debugger on subprocess (could not resolve filename nor module name).')
            return original_args
        from _pydevd_bundle.pydevd_command_line_handling import setup_to_argv
        new_args = []
        new_args.extend(unquoted_args[:first_non_vm_index])
        if before_module_flag:
            new_args.append(before_module_flag)
        add_module_at = len(new_args) + 1
        new_args.extend(setup_to_argv(_get_setup_updated_with_protocol_and_ppid(SetupHolder.setup, is_exec=is_exec), skip_names=set(('module', 'cmd-line'))))
        new_args.append('--file')
        if module_name is not None:
            assert module_name_i_start != -1
            assert module_name_i_end != -1
            new_args.insert(add_module_at, '--module')
            new_args.append(module_name)
            new_args.extend(unquoted_args[module_name_i_end + 1:])
        elif filename is not None:
            assert filename_i != -1
            new_args.append(filename)
            new_args.extend(unquoted_args[filename_i + 1:])
        else:
            raise AssertionError('Internal error (unexpected condition)')
        return quote_args(new_args)
    except:
        pydev_log.exception('Error patching args (debugger not attached to subprocess).')
        return original_args