from paste import httpexceptions
from paste.util import converters
import tempfile
from io import BytesIO
def make_cascade(loader, global_conf, catch='404', **local_conf):
    """
    Entry point for Paste Deploy configuration

    Expects configuration like::

        [composit:cascade]
        use = egg:Paste#cascade
        # all start with 'app' and are sorted alphabetically
        app1 = foo
        app2 = bar
        ...
        catch = 404 500 ...
    """
    catch = map(int, converters.aslist(catch))
    apps = []
    for name, value in local_conf.items():
        if not name.startswith('app'):
            raise ValueError("Bad configuration key %r (=%r); all configuration keys must start with 'app'" % (name, value))
        app = loader.get_app(value, global_conf=global_conf)
        apps.append((name, app))
    apps.sort()
    apps = [app for name, app in apps]
    return Cascade(apps, catch=catch)