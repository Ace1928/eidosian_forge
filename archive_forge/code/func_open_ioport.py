import importlib
import os
from .. import ports
def open_ioport(self, name=None, virtual=False, callback=None, autoreset=False, **kwargs):
    """Open a port for input and output.

        If the environment variable MIDO_DEFAULT_IOPORT is set,
        it will override the default port.

        virtual=False
          Passing True opens a new port that other applications can
          connect to. Raises IOError if not supported by the backend.

        callback=None
          A callback function to be called when a new message arrives.
          The function should take one argument (the message).
          Raises IOError if not supported by the backend.

        autoreset=False
          Automatically send all_notes_off and reset_all_controllers
          on all channels. This is the same as calling `port.reset()`.
        """
    kwargs.update(dict(virtual=virtual, callback=callback, autoreset=autoreset))
    if name is None:
        name = self._env('MIDO_DEFAULT_IOPORT') or None
    if hasattr(self.module, 'IOPort'):
        return self.module.IOPort(name, **self._add_api(kwargs))
    else:
        if name:
            input_name = output_name = name
        else:
            input_name = self._env('MIDO_DEFAULT_INPUT')
            output_name = self._env('MIDO_DEFAULT_OUTPUT')
        kwargs = self._add_api(kwargs)
        return ports.IOPort(self.module.Input(input_name, **kwargs), self.module.Output(output_name, **kwargs))