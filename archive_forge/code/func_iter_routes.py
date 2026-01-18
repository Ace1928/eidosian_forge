import weakref
import importlib_metadata
from wsme.exc import ClientSideError
def iter_routes(self):
    for attrname in dir(self):
        attr = getattr(self, attrname)
        if getattr(attr, 'exposed', False):
            for path in _cfg(attr)['paths']:
                yield (self.resolve_path(path), attr)