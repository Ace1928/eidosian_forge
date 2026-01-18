import os
import functools
def config_environment():
    global SetHostMTurkConnection
    try:
        local = os.path.join(os.path.dirname(__file__), 'local.py')
        execfile(local)
    except:
        pass
    if live_connection:
        from boto.mturk.connection import MTurkConnection
    else:
        os.environ.setdefault('AWS_ACCESS_KEY_ID', 'foo')
        os.environ.setdefault('AWS_SECRET_ACCESS_KEY', 'bar')
        from mocks import MTurkConnection
    SetHostMTurkConnection = functools.partial(MTurkConnection, host=mturk_host)