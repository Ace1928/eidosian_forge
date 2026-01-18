import os
import warnings
import builtins
import cherrypy
def check_config_namespaces(self):
    """Process config and warn on each unknown config namespace."""
    for sn, app in cherrypy.tree.apps.items():
        if not isinstance(app, cherrypy.Application):
            continue
        self._known_ns(app)