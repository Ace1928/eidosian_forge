import abc
import enum
from taskflow import atom
from taskflow import exceptions as exc
from taskflow.utils import misc
def outcomes_iter(self, index=None):
    """Iterates over the contained failure outcomes.

        If the index is not provided, then all outcomes are iterated over.

        NOTE(harlowja): if the retry itself failed, this will **not** include
        those types of failures. Use the :py:attr:`.failure` attribute to
        access that instead (if it exists, aka, non-none).
        """
    if index is None:
        contents = self._contents
    else:
        contents = [self._contents[index]]
    for provided, outcomes in contents:
        for owner, outcome in outcomes.items():
            yield (owner, outcome)