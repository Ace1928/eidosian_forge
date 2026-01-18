import os
import warnings
import builtins
import cherrypy
def check_config_types(self):
    """Assert that config values are of the same type as default values."""
    self._known_types(cherrypy.config)
    for sn, app in cherrypy.tree.apps.items():
        if not isinstance(app, cherrypy.Application):
            continue
        self._known_types(app.config)