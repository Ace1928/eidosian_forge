import os
import re
import sys
def get_platform_osx(_config_vars, osname, release, machine):
    """Filter values for get_platform()"""
    macver = _config_vars.get('MACOSX_DEPLOYMENT_TARGET', '')
    if macver and '.' not in macver:
        macver += '.0'
    macrelease = _get_system_version() or macver
    macver = macver or macrelease
    if macver:
        release = macver
        osname = 'macosx'
        cflags = _config_vars.get(_INITPRE + 'CFLAGS', _config_vars.get('CFLAGS', ''))
        if macrelease:
            try:
                macrelease = tuple((int(i) for i in macrelease.split('.')[0:2]))
            except ValueError:
                macrelease = (10, 3)
        else:
            macrelease = (10, 3)
        if macrelease >= (10, 4) and '-arch' in cflags.strip():
            machine = 'fat'
            archs = re.findall('-arch\\s+(\\S+)', cflags)
            archs = tuple(sorted(set(archs)))
            if len(archs) == 1:
                machine = archs[0]
            elif archs == ('arm64', 'x86_64'):
                machine = 'universal2'
            elif archs == ('i386', 'ppc'):
                machine = 'fat'
            elif archs == ('i386', 'x86_64'):
                machine = 'intel'
            elif archs == ('i386', 'ppc', 'x86_64'):
                machine = 'fat3'
            elif archs == ('ppc64', 'x86_64'):
                machine = 'fat64'
            elif archs == ('i386', 'ppc', 'ppc64', 'x86_64'):
                machine = 'universal'
            else:
                raise ValueError("Don't know machine value for archs=%r" % (archs,))
        elif machine == 'i386':
            if sys.maxsize >= 2 ** 32:
                machine = 'x86_64'
        elif machine in ('PowerPC', 'Power_Macintosh'):
            if sys.maxsize >= 2 ** 32:
                machine = 'ppc64'
            else:
                machine = 'ppc'
    return (osname, release, machine)