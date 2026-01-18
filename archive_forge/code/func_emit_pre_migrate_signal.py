import sys
from django.apps import apps
from django.db import models
def emit_pre_migrate_signal(verbosity, interactive, db, **kwargs):
    for app_config in apps.get_app_configs():
        if app_config.models_module is None:
            continue
        if verbosity >= 2:
            stdout = kwargs.get('stdout', sys.stdout)
            stdout.write('Running pre-migrate handlers for application %s' % app_config.label)
        models.signals.pre_migrate.send(sender=app_config, app_config=app_config, verbosity=verbosity, interactive=interactive, using=db, **kwargs)