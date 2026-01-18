import os.path
from .exceptions import NoSuchWidgetError, WidgetPluginError
def createQObject(self, classname, *args, **kwargs):
    factory = self.findQObjectType(classname)
    if factory is None:
        parts = classname.split('.')
        if len(parts) > 1:
            factory = self.findQObjectType(parts[0])
            if factory is not None:
                for part in parts[1:]:
                    factory = getattr(factory, part, None)
                    if factory is None:
                        break
        if factory is None:
            raise NoSuchWidgetError(classname)
    return self._cpolicy.instantiate(factory, *args, **kwargs)