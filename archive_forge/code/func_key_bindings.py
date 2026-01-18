from __future__ import unicode_literals
from abc import ABCMeta, abstractmethod
from prompt_toolkit.cache import SimpleCache
from prompt_toolkit.filters import CLIFilter, to_cli_filter, Never
from prompt_toolkit.keys import Key, Keys
from six import text_type, with_metaclass
@property
def key_bindings(self):
    self._update_cache()
    return self._registry2.key_bindings