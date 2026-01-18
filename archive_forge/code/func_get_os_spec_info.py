import json
import locale
import multiprocessing
import os
import platform
import textwrap
import sys
from contextlib import redirect_stdout
from datetime import datetime
from io import StringIO
from subprocess import check_output, PIPE, CalledProcessError
import numpy as np
import llvmlite.binding as llvmbind
from llvmlite import __version__ as llvmlite_version
from numba import cuda as cu, __version__ as version_number
from numba.cuda import cudadrv
from numba.cuda.cudadrv.driver import driver as cudriver
from numba.cuda.cudadrv.runtime import runtime as curuntime
from numba.core import config
def get_os_spec_info(os_name):

    class CmdBufferOut(tuple):
        buffer_output_flag = True

    class CmdReadFile(tuple):
        read_file_flag = True
    shell_params = {'Linux': {'cmd': (CmdReadFile(('/sys/fs/cgroup/cpuacct/cpu.cfs_quota_us',)), CmdReadFile(('/sys/fs/cgroup/cpuacct/cpu.cfs_period_us',))), 'cmd_optional': (CmdReadFile(('/proc/meminfo',)), CmdReadFile(('/proc/self/status',))), 'kwds': {'MemTotal:': _mem_total, 'MemAvailable:': _mem_available, 'Cpus_allowed:': _cpus_allowed, 'Cpus_allowed_list:': _cpus_list, '/sys/fs/cgroup/cpuacct/cpu.cfs_quota_us': _cfs_quota, '/sys/fs/cgroup/cpuacct/cpu.cfs_period_us': _cfs_period}}, 'Windows': {'cmd': (), 'cmd_optional': (CmdBufferOut(('wmic', 'OS', 'get', 'TotalVirtualMemorySize')), CmdBufferOut(('wmic', 'OS', 'get', 'FreeVirtualMemory'))), 'kwds': {'TotalVirtualMemorySize': _mem_total, 'FreeVirtualMemory': _mem_available}}, 'Darwin': {'cmd': (), 'cmd_optional': (('sysctl', 'hw.memsize'), 'vm_stat'), 'kwds': {'hw.memsize:': _mem_total, 'free:': _mem_available}, 'units': {_mem_total: 1, _mem_available: 4096}}}
    os_spec_info = {}
    params = shell_params.get(os_name, {})
    cmd_selected = params.get('cmd', ())
    if _psutil_import:
        vm = psutil.virtual_memory()
        os_spec_info.update({_mem_total: vm.total, _mem_available: vm.available})
        p = psutil.Process()
        cpus_allowed = p.cpu_affinity() if hasattr(p, 'cpu_affinity') else []
        if cpus_allowed:
            os_spec_info[_cpus_allowed] = len(cpus_allowed)
            os_spec_info[_cpus_list] = ' '.join((str(n) for n in cpus_allowed))
    else:
        _warning_log.append('Warning (psutil): psutil cannot be imported. For more accuracy, consider installing it.')
        cmd_selected += params.get('cmd_optional', ())
    output = []
    for cmd in cmd_selected:
        if hasattr(cmd, 'read_file_flag'):
            if os.path.exists(cmd[0]):
                try:
                    with open(cmd[0], 'r') as f:
                        out = f.readlines()
                        if out:
                            out[0] = ' '.join((cmd[0], out[0]))
                            output.extend(out)
                except OSError as e:
                    _error_log.append(f'Error (file read): {e}')
                    continue
            else:
                _warning_log.append('Warning (no file): {}'.format(cmd[0]))
                continue
        else:
            try:
                out = check_output(cmd, stderr=PIPE)
            except (OSError, CalledProcessError) as e:
                _error_log.append(f'Error (subprocess): {e}')
                continue
            if hasattr(cmd, 'buffer_output_flag'):
                out = b' '.join((line for line in out.splitlines())) + b'\n'
            output.extend(out.decode().splitlines())
    kwds = params.get('kwds', {})
    for line in output:
        match = kwds.keys() & line.split()
        if match and len(match) == 1:
            k = kwds[match.pop()]
            os_spec_info[k] = line
        elif len(match) > 1:
            print(f'Ambiguous output: {line}')

    def format():
        split = os_spec_info.get(_cfs_quota, '').split()
        if split:
            os_spec_info[_cfs_quota] = float(split[-1])
        split = os_spec_info.get(_cfs_period, '').split()
        if split:
            os_spec_info[_cfs_period] = float(split[-1])
        if os_spec_info.get(_cfs_quota, -1) != -1:
            cfs_quota = os_spec_info.get(_cfs_quota, '')
            cfs_period = os_spec_info.get(_cfs_period, '')
            runtime_amount = cfs_quota / cfs_period
            os_spec_info[_cfs_restrict] = runtime_amount

    def format_optional():
        units = {_mem_total: 1024, _mem_available: 1024}
        units.update(params.get('units', {}))
        for k in (_mem_total, _mem_available):
            digits = ''.join((d for d in os_spec_info.get(k, '') if d.isdigit()))
            os_spec_info[k] = int(digits or 0) * units[k]
        split = os_spec_info.get(_cpus_allowed, '').split()
        if split:
            n = split[-1]
            n = n.split(',')[-1]
            os_spec_info[_cpus_allowed] = str(bin(int(n or 0, 16))).count('1')
        split = os_spec_info.get(_cpus_list, '').split()
        if split:
            os_spec_info[_cpus_list] = split[-1]
    try:
        format()
        if not _psutil_import:
            format_optional()
    except Exception as e:
        _error_log.append(f'Error (format shell output): {e}')
    os_specific_funcs = {'Linux': {_libc_version: lambda: ' '.join(platform.libc_ver())}, 'Windows': {_os_spec_version: lambda: ' '.join((s for s in platform.win32_ver()))}, 'Darwin': {_os_spec_version: lambda: ''.join((i or ' ' for s in tuple(platform.mac_ver()) for i in s))}}
    key_func = os_specific_funcs.get(os_name, {})
    os_spec_info.update({k: f() for k, f in key_func.items()})
    return os_spec_info