from __future__ import absolute_import, division, print_function
import os
import tempfile
from ansible.module_utils.basic import AnsibleModule
def create_admin_file():
    desc, filename = tempfile.mkstemp(prefix='ansible_svr4pkg', text=True)
    fullauto = b'\nmail=\ninstance=unique\npartial=nocheck\nrunlevel=quit\nidepend=nocheck\nrdepend=nocheck\nspace=quit\nsetuid=nocheck\nconflict=nocheck\naction=nocheck\nnetworktimeout=60\nnetworkretries=3\nauthentication=quit\nkeystore=/var/sadm/security\nproxy=\nbasedir=default\n'
    os.write(desc, fullauto)
    os.close(desc)
    return filename