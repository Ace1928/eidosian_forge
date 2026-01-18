import inspect, os, sys, textwrap
from IPython.core.error import UsageError
from IPython.core.magic import Magics, magics_class, line_magic
from IPython.testing.skipdoctest import skip_doctest
from traitlets import Bool
def refresh_variables(ip):
    db = ip.db
    for key in db.keys('autorestore/*'):
        justkey = os.path.basename(key)
        try:
            obj = db[key]
        except KeyError:
            print("Unable to restore variable '%s', ignoring (use %%store -d to forget!)" % justkey)
            print('The error was:', sys.exc_info()[0])
        else:
            ip.user_ns[justkey] = obj