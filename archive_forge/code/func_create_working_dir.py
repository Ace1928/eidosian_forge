import os
import boto
from boto.utils import get_instance_metadata, get_instance_userdata
from boto.pyami.config import Config, BotoConfigPath
from boto.pyami.scriptbase import ScriptBase
import time
def create_working_dir(self):
    boto.log.info('Working directory: %s' % self.working_dir)
    if not os.path.exists(self.working_dir):
        os.mkdir(self.working_dir)