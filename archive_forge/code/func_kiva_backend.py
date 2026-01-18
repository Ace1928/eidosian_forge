import sys
import os
from os import path
from contextlib import contextmanager
@property
def kiva_backend(self):
    """
        Property getter for the Kiva backend. The value returned is dependent
        on the value of the toolkit property. If toolkit specifies a kiva
        backend using the extended syntax: <enable toolkit>[.<kiva backend>]
        then the value of the property will be whatever was specified.
        Otherwise the value will be a reasonable default for the given enable
        backend.
        """
    if self._toolkit is None:
        raise AttributeError('The kiva_backend attribute is dependent on toolkit, which has not been set.')
    if self._kiva_backend is None:
        try:
            self._kiva_backend = self._toolkit.split('.')[1]
        except IndexError:
            if self.toolkit == 'wx':
                self._kiva_backend = 'quartz' if sys.platform == 'darwin' else 'image'
            elif self.toolkit in ['qt4', 'qt']:
                self._kiva_backend = 'image'
            else:
                self._kiva_backend = 'image'
    return self._kiva_backend