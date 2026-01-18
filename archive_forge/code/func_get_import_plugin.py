from oslo_config import cfg
from stevedore import named
from glance.i18n import _
import_filtering_opts = [
def get_import_plugin(**kwargs):
    method_list = CONF.enabled_import_methods
    import_method = kwargs.get('import_req')['method']['name']
    if import_method in method_list:
        import_method = import_method.replace('-', '_')
        task_list = [import_method]
    extensions = named.NamedExtensionManager('glance.image_import.internal_plugins', names=task_list, name_order=True, invoke_on_load=True, invoke_kwds=kwargs)
    for extension in extensions.extensions:
        return extension.obj