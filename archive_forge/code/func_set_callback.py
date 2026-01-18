import collections
import json
import os
from tensorflow.python.debug.cli import debugger_cli_common
from tensorflow.python.platform import gfile
def set_callback(self, property_name, callback):
    """Set a set-callback for given property.

    Args:
      property_name: Name of the property.
      callback: The callback as a `callable` of signature:
          def cbk(config):
        where config is the config after it is set to the new value.
        The callback is invoked each time the set() method is called with the
        matching property_name.

    Raises:
      KeyError: If property_name does not exist.
      TypeError: If `callback` is not callable.
    """
    if property_name not in self._config:
        raise KeyError('%s is not a valid property name.' % property_name)
    if not callable(callback):
        raise TypeError('The callback object provided is not callable.')
    self._set_callbacks[property_name] = callback