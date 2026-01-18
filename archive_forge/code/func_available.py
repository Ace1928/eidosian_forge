import importlib
@property
def available(self):
    if self._available is None:
        try:
            self.initialize()
            self._available = True
        except ImportError:
            self._available = False
    return self._available