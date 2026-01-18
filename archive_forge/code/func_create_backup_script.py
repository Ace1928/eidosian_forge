import boto
from boto.manage.volume import Volume
from boto.exception import EC2ResponseError
import os, time
from boto.pyami.installers.ubuntu.installer import Installer
from string import Template
import boto
from boto.pyami.scriptbase import ScriptBase
import traceback
import boto
from boto.manage.volume import Volume
import boto
def create_backup_script(self):
    t = Template(BackupScriptTemplate)
    s = t.substitute(volume_id=self.volume_id, instance_id=self.instance_id, mount_point=self.mount_point)
    fp = open('/usr/local/bin/ebs_backup', 'w')
    fp.write(s)
    fp.close()
    self.run('chmod +x /usr/local/bin/ebs_backup')