from __future__ import annotations
from collections import abc
from typing import TYPE_CHECKING, Any, Mapping, Optional, Sequence
from pymongo import max_staleness_selectors
from pymongo.errors import ConfigurationError
from pymongo.server_selectors import (
@property
def hedge(self) -> Optional[_Hedge]:
    """The read preference ``hedge`` parameter.

        A dictionary that configures how the server will perform hedged reads.
        It consists of the following keys:

        - ``enabled``: Enables or disables hedged reads in sharded clusters.

        Hedged reads are automatically enabled in MongoDB 4.4+ when using a
        ``nearest`` read preference. To explicitly enable hedged reads, set
        the ``enabled`` key  to ``true``::

            >>> Nearest(hedge={'enabled': True})

        To explicitly disable hedged reads, set the ``enabled`` key  to
        ``False``::

            >>> Nearest(hedge={'enabled': False})

        .. versionadded:: 3.11
        """
    return self.__hedge