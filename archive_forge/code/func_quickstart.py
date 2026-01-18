from threading import local as _local
from ._cperror import (
from . import _cpdispatch as dispatch
from ._cptools import default_toolbox as tools, Tool
from ._helper import expose, popargs, url
from . import _cprequest, _cpserver, _cptree, _cplogging, _cpconfig
import cherrypy.lib.httputil as _httputil
from ._cptree import Application
from . import _cpwsgi as wsgi
from . import process
from . import _cpchecker
def quickstart(root=None, script_name='', config=None):
    """Mount the given root, start the builtin server (and engine), then block.

    root: an instance of a "controller class" (a collection of page handler
        methods) which represents the root of the application.
    script_name: a string containing the "mount point" of the application.
        This should start with a slash, and be the path portion of the URL
        at which to mount the given root. For example, if root.index() will
        handle requests to "http://www.example.com:8080/dept/app1/", then
        the script_name argument would be "/dept/app1".

        It MUST NOT end in a slash. If the script_name refers to the root
        of the URI, it MUST be an empty string (not "/").
    config: a file or dict containing application config. If this contains
        a [global] section, those entries will be used in the global
        (site-wide) config.
    """
    if config:
        _global_conf_alias.update(config)
    tree.mount(root, script_name, config)
    engine.signals.subscribe()
    engine.start()
    engine.block()