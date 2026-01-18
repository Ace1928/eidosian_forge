from __future__ import annotations
import sys, os, subprocess, shutil
import shlex
import typing as T
from .. import envconfig
from .. import mlog
from ..compilers import compilers
from ..compilers.detect import defaults as compiler_names
def detect_cross_debianlike(options: T.Any) -> MachineInfo:
    if options.debarch == 'auto':
        cmd = ['dpkg-architecture']
    else:
        cmd = ['dpkg-architecture', '-a' + options.debarch]
    output = subprocess.check_output(cmd, universal_newlines=True, stderr=subprocess.DEVNULL)
    data = {}
    for line in output.split('\n'):
        line = line.strip()
        if line == '':
            continue
        k, v = line.split('=', 1)
        data[k] = v
    host_arch = data['DEB_HOST_GNU_TYPE']
    host_os = data['DEB_HOST_ARCH_OS']
    host_subsystem = host_os
    host_kernel = 'linux'
    host_cpu_family = deb_cpu_family_map.get(data['DEB_HOST_GNU_CPU'], data['DEB_HOST_GNU_CPU'])
    host_cpu = deb_cpu_map.get(data['DEB_HOST_ARCH'], data['DEB_HOST_ARCH'])
    host_endian = data['DEB_HOST_ARCH_ENDIAN']
    compilerstems = [('c', 'gcc'), ('cpp', 'g++'), ('objc', 'gobjc'), ('objcpp', 'gobjc++')]
    infos = MachineInfo()
    deb_compiler_lookup(infos, compilerstems, host_arch, options.gccsuffix)
    if len(infos.compilers) == 0:
        print('Warning: no compilers were detected.')
    infos.binaries['ar'] = locate_path('%s-ar' % host_arch)
    infos.binaries['strip'] = locate_path('%s-strip' % host_arch)
    infos.binaries['objcopy'] = locate_path('%s-objcopy' % host_arch)
    infos.binaries['ld'] = locate_path('%s-ld' % host_arch)
    try:
        infos.binaries['cmake'] = locate_path('cmake')
        deb_detect_cmake(infos, data)
    except ValueError:
        pass
    try:
        infos.binaries['pkg-config'] = locate_path('%s-pkg-config' % host_arch)
    except ValueError:
        pass
    try:
        infos.binaries['cups-config'] = locate_path('cups-config')
    except ValueError:
        pass
    infos.system = host_os
    infos.subsystem = host_subsystem
    infos.kernel = host_kernel
    infos.cpu_family = host_cpu_family
    infos.cpu = host_cpu
    infos.endian = host_endian
    get_args_from_envvars(infos)
    return infos