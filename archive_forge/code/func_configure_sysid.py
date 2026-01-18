from __future__ import absolute_import, division, print_function
import os
import platform
import re
import tempfile
import time
from ansible.module_utils.basic import AnsibleModule
def configure_sysid(self):
    if os.path.isfile('%s/root/etc/.UNCONFIGURED' % self.path):
        os.unlink('%s/root/etc/.UNCONFIGURED' % self.path)
    open('%s/root/noautoshutdown' % self.path, 'w').close()
    node = open('%s/root/etc/nodename' % self.path, 'w')
    node.write(self.name)
    node.close()
    id = open('%s/root/etc/.sysIDtool.state' % self.path, 'w')
    id.write('1       # System previously configured?\n')
    id.write('1       # Bootparams succeeded?\n')
    id.write('1       # System is on a network?\n')
    id.write('1       # Extended network information gathered?\n')
    id.write('0       # Autobinder succeeded?\n')
    id.write('1       # Network has subnets?\n')
    id.write('1       # root password prompted for?\n')
    id.write('1       # locale and term prompted for?\n')
    id.write('1       # security policy in place\n')
    id.write('1       # NFSv4 domain configured\n')
    id.write('0       # Auto Registration Configured\n')
    id.write('vt100')
    id.close()