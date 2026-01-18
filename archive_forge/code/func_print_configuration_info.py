import argparse
import os
import platform
import subprocess
import sys
import time
def print_configuration_info():
    print('Test configuration:')
    if sys.platform == 'darwin':
        sys.path.append(os.path.abspath('test/lib'))
        import TestMac
        print(f'  Mac {platform.mac_ver()[0]} {platform.mac_ver()[2]}')
        print(f'  Xcode {TestMac.Xcode.Version()}')
    elif sys.platform == 'win32':
        sys.path.append(os.path.abspath('pylib'))
        import gyp.MSVSVersion
        print('  Win %s %s\n' % platform.win32_ver()[0:2])
        print('  MSVS %s' % gyp.MSVSVersion.SelectVisualStudioVersion().Description())
    elif sys.platform in ('linux', 'linux2'):
        print('  Linux %s' % ' '.join(platform.linux_distribution()))
    print(f'  Python {platform.python_version()}')
    print(f'  PYTHONPATH={os.environ['PYTHONPATH']}')
    print()