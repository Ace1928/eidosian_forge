import importlib
import os
from .. import ports
def open_input(self, name=None, virtual=False, callback=None, **kwargs):
    """Open an input port.

        If the environment variable MIDO_DEFAULT_INPUT is set,
        it will override the default port.

        virtual=False
          Passing True opens a new port that other applications can
          connect to. Raises IOError if not supported by the backend.

        callback=None
          A callback function to be called when a new message arrives.
          The function should take one argument (the message).
          Raises IOError if not supported by the backend.
        """
    kwargs.update(dict(virtual=virtual, callback=callback))
    if name is None:
        name = self._env('MIDO_DEFAULT_INPUT')
    return self.module.Input(name, **self._add_api(kwargs))