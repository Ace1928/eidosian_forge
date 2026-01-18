import os
import re
import warnings
import boto
from boto.compat import expanduser, ConfigParser, NoOptionError, NoSectionError, StringIO
def dump_safe(self, fp=None):
    if not fp:
        fp = StringIO()
    for section in self.sections():
        fp.write('[%s]\n' % section)
        for option in self.options(section):
            if option == 'aws_secret_access_key':
                fp.write('%s = xxxxxxxxxxxxxxxxxx\n' % option)
            else:
                fp.write('%s = %s\n' % (option, self.get(section, option)))