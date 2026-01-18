import os
import re
import warnings
import boto
from boto.compat import expanduser, ConfigParser, NoOptionError, NoSectionError, StringIO
def save_option(self, path, section, option, value):
    """
        Write the specified Section.Option to the config file specified by path.
        Replace any previous value.  If the path doesn't exist, create it.
        Also add the option the the in-memory config.
        """
    config = ConfigParser()
    config.read(path)
    if not config.has_section(section):
        config.add_section(section)
    config.set(section, option, value)
    fp = open(path, 'w')
    config.write(fp)
    fp.close()
    if not self.has_section(section):
        self.add_section(section)
    self.set(section, option, value)