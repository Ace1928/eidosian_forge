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
def create_backup_cleanup_script(self, use_tag_based_cleanup=False):
    fp = open('/usr/local/bin/ebs_backup_cleanup', 'w')
    if use_tag_based_cleanup:
        fp.write(TagBasedBackupCleanupScript)
    else:
        fp.write(BackupCleanupScript)
    fp.close()
    self.run('chmod +x /usr/local/bin/ebs_backup_cleanup')