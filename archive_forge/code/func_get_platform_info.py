from __future__ import (absolute_import, division, print_function)
import json
import platform
import io
import os
def get_platform_info():
    result = dict(platform_dist_result=[])
    if hasattr(platform, 'dist'):
        result['platform_dist_result'] = platform.dist()
    osrelease_content = read_utf8_file('/etc/os-release')
    if not osrelease_content:
        osrelease_content = read_utf8_file('/usr/lib/os-release')
    result['osrelease_content'] = osrelease_content
    return result