import os.path
from .exceptions import NoSuchWidgetError, WidgetPluginError
def findQObjectType(self, classname):
    for module in self._modules:
        w = module.search(classname)
        if w is not None:
            return w
    return None