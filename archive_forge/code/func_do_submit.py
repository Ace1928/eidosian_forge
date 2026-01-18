from optparse import OptionParser
from boto.services.servicedef import ServiceDef
from boto.services.submit import Submitter
from boto.services.result import ResultProcessor
import boto
import sys, os
from boto.compat import StringIO
def do_submit(self):
    if not self.options.path:
        self.parser.error('No path provided')
    if not os.path.exists(self.options.path):
        self.parser.error('Invalid path (%s)' % self.options.path)
    s = Submitter(self.sd)
    t = s.submit_path(self.options.path, None, self.options.ignore, None, None, True, self.options.path)
    print('A total of %d files were submitted' % t[1])
    print('Batch Identifier: %s' % t[0])