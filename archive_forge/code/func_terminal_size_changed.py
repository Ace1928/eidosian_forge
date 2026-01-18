from __future__ import unicode_literals
from abc import ABCMeta, abstractmethod
from six import with_metaclass
@abstractmethod
def terminal_size_changed(self):
    pass