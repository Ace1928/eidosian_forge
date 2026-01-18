@property
def tensor(self):
    return self._tensor() if callable(self._tensor) else self._tensor