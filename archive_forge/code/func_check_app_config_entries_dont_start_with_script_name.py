import os
import warnings
import builtins
import cherrypy
def check_app_config_entries_dont_start_with_script_name(self):
    """Check for App config with sections that repeat script_name."""
    for sn, app in cherrypy.tree.apps.items():
        if not isinstance(app, cherrypy.Application):
            continue
        if not app.config:
            continue
        if sn == '':
            continue
        sn_atoms = sn.strip('/').split('/')
        for key in app.config.keys():
            key_atoms = key.strip('/').split('/')
            if key_atoms[:len(sn_atoms)] == sn_atoms:
                warnings.warn('The application mounted at %r has config entries that start with its script name: %r' % (sn, key))