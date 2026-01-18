from __future__ import annotations
from typing import Generator
from typing import NamedTuple
from flake8.violation import Violation
def statistics_for(self, prefix: str, filename: str | None=None) -> Generator[Statistic, None, None]:
    """Generate statistics for the prefix and filename.

        If you have a :class:`Statistics` object that has recorded errors,
        you can generate the statistics for a prefix (e.g., ``E``, ``E1``,
        ``W50``, ``W503``) with the optional filter of a filename as well.

        .. code-block:: python

            >>> stats = Statistics()
            >>> stats.statistics_for('E12',
                                     filename='src/flake8/statistics.py')
            <generator ...>
            >>> stats.statistics_for('W')
            <generator ...>

        :param prefix:
            The error class or specific error code to find statistics for.
        :param filename:
            (Optional) The filename to further filter results by.
        :returns:
            Generator of instances of :class:`Statistic`
        """
    matching_errors = sorted((key for key in self._store if key.matches(prefix, filename)))
    for error_code in matching_errors:
        yield self._store[error_code]