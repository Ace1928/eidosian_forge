import pkgutil
import sys
from _pydev_bundle import pydev_log
def get_extension_classes(self, extension_type):
    self._ensure_loaded()
    if extension_type in self.type_to_instance:
        return self.type_to_instance[extension_type]
    handlers = self.type_to_instance.setdefault(extension_type, [])
    for attr_name, attr in self._iter_attr():
        if isinstance(attr, type) and issubclass(attr, extension_type) and (attr is not extension_type):
            try:
                handlers.append(attr())
            except:
                pydev_log.exception('Unable to load extension class: %s', attr_name)
    return handlers