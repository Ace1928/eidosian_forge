import os
from pecan import load_app
def reset_global_config():
    """
    When tests alter application configurations they can get sticky and pollute
    other tests that might rely on a pristine configuration. This helper will
    reset the config by overwriting it with ``pecan.configuration.DEFAULT``.
    """
    from pecan import configuration
    configuration.set_config(dict(configuration.initconf()), overwrite=True)
    os.environ.pop('PECAN_CONFIG', None)