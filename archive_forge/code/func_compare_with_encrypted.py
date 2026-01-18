from __future__ import absolute_import, division, print_function
import pytest
from awx.main.models import NotificationTemplate, Job
def compare_with_encrypted(model_config, param_config):
    """Given a model_config from the database, assure that this is consistent
    with the config given in the notification_configuration parameter
    this requires handling of password fields
    """
    for key, model_val in model_config.items():
        param_val = param_config.get(key, 'missing')
        if isinstance(model_val, str) and (model_val.startswith('$encrypted$') or param_val.startswith('$encrypted$')):
            assert model_val.startswith('$encrypted$')
            assert len(model_val) > len('$encrypted$')
        else:
            assert model_val == param_val, 'Config key {0} did not match, (model: {1}, input: {2})'.format(key, model_val, param_val)