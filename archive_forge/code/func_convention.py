import abc
from yaql.language import exceptions
from yaql.language import runner
from yaql.language import specs
from yaql.language import utils
@property
def convention(self):
    return self._convention