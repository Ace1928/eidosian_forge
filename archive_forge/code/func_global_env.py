from stevedore import extension
from heat.common import pluginutils
from heat.engine import clients
from heat.engine import environment
from heat.engine import plugin_manager
def global_env():
    if _environment is None:
        initialise()
    return _environment