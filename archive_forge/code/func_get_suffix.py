from __future__ import unicode_literals
from abc import ABCMeta, abstractmethod
from six import with_metaclass
def get_suffix(completion):
    return completion.text[-completion.start_position:]