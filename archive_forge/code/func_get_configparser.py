importing any other Kivy modules. Ideally, this means setting them right at
from collections import OrderedDict
from os import environ
from os.path import exists
from weakref import ref
from kivy import kivy_config_fn
from kivy.compat import PY2, string_types
from kivy.logger import Logger, logger_config_update
from kivy.utils import platform
@staticmethod
def get_configparser(name):
    """Returns the :class:`ConfigParser` instance whose name is `name`, or
        None if not found.

        :Parameters:
            `name`: string
                The name of the :class:`ConfigParser` instance to return.
        """
    try:
        config = ConfigParser._named_configs[name][0]
        if config is not None:
            config = config()
            if config is not None:
                return config
        del ConfigParser._named_configs[name]
    except KeyError:
        return None