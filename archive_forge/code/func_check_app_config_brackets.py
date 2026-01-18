import os
import warnings
import builtins
import cherrypy
def check_app_config_brackets(self):
    """Check for App config with extraneous brackets in section names."""
    for sn, app in cherrypy.tree.apps.items():
        if not isinstance(app, cherrypy.Application):
            continue
        if not app.config:
            continue
        for key in app.config.keys():
            if key.startswith('[') or key.endswith(']'):
                warnings.warn('The application mounted at %r has config section names with extraneous brackets: %r. Config *files* need brackets; config *dicts* (e.g. passed to tree.mount) do not.' % (sn, key))