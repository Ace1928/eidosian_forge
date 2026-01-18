from __future__ import absolute_import
from __future__ import print_function
import sys
from ruamel.yaml.error import YAMLError, YAMLStreamError
from ruamel.yaml.events import *  # NOQA
from ruamel.yaml.compat import utf8, text_type, PY2, nprint, dbg, DBG_EVENT, \
def prepare_version(self, version):
    major, minor = version
    if major != 1:
        raise EmitterError('unsupported YAML version: %d.%d' % (major, minor))
    return u'%d.%d' % (major, minor)