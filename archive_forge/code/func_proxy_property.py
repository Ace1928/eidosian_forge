import functools
def proxy_property(self):
    return getattr(self._file, attr_name)