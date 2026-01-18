import types
import os
import string
import uuid
from paste.deploy import appconfig
from paste.script import copydir
from paste.script.command import Command, BadCommand, run as run_command
from paste.script.util import secret
from paste.util import import_string
import paste.script.templates
import pkg_resources
def run_editor(self):
    filenames = self.installer.editable_config_files(self.config_file)
    if filenames is None:
        print('Warning: the config file is not known (--edit ignored)')
        return False
    if not filenames:
        print('Warning: no config files need editing (--edit ignored)')
        return True
    if len(filenames) > 1:
        print('Warning: there is more than one editable config file (--edit ignored)')
        return False
    if not os.environ.get('EDITOR'):
        print('Error: you must set $EDITOR if using --edit')
        return False
    if self.verbose:
        print('%s %s' % (os.environ['EDITOR'], filenames[0]))
    retval = os.system('$EDITOR %s' % filenames[0])
    if retval:
        print('Warning: editor %s returned with error code %i' % (os.environ['EDITOR'], retval))
        return False
    return True