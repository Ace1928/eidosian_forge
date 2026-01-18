import inspect, os, sys, textwrap
from IPython.core.error import UsageError
from IPython.core.magic import Magics, magics_class, line_magic
from IPython.testing.skipdoctest import skip_doctest
from traitlets import Bool
def restore_dhist(ip):
    ip.user_ns['_dh'] = ip.db.get('dhist', [])