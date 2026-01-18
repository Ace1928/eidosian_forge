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
def handle_mount_point(self):
    boto.log.info('handle_mount_point')
    if not os.path.isdir(self.mount_point):
        boto.log.info('making directory')
        self.run('mkdir %s' % self.mount_point)
    else:
        boto.log.info('directory exists already')
        self.run('mount -l')
        lines = self.last_command.output.split('\n')
        for line in lines:
            t = line.split()
            if t and t[2] == self.mount_point:
                if t[0] != self.device:
                    self.run('umount %s' % self.mount_point)
                    self.run('mount %s /tmp' % t[0])
                    break
    self.run('chmod 777 /tmp')
    self.run('mount %s %s' % (self.device, self.mount_point))
    self.run('xfs_growfs %s' % self.mount_point)