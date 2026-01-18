from __future__ import unicode_literals
from abc import ABCMeta, abstractmethod
from six import with_metaclass
from .filters import to_cli_filter
def get_suggestion(self, cli, buffer, document):
    if self.filter(cli):
        return self.auto_suggest.get_suggestion(cli, buffer, document)