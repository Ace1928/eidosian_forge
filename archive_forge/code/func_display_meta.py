from __future__ import unicode_literals
from abc import ABCMeta, abstractmethod
from six import with_metaclass
@property
def display_meta(self):
    if self._display_meta is not None:
        return self._display_meta
    elif self._get_display_meta:
        self._display_meta = self._get_display_meta()
        return self._display_meta
    else:
        return ''