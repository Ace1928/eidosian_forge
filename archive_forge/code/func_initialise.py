import weakref
from oslo_config import cfg
from oslo_log import log as logging
from oslo_utils import importutils
from stevedore import enabled
from heat.common import exception
from heat.common.i18n import _
from heat.common import pluginutils
def initialise():
    global _mgr
    if _mgr:
        return

    def client_is_available(client_plugin):
        if not hasattr(client_plugin.plugin, 'is_available'):
            return True
        return client_plugin.plugin.is_available()
    _mgr = enabled.EnabledExtensionManager(namespace='heat.clients', check_func=client_is_available, invoke_on_load=False, on_load_failure_callback=pluginutils.log_fail_msg)