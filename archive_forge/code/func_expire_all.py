from __future__ import annotations
from typing import Any
from typing import Callable
from typing import Generic
from typing import Iterable
from typing import Iterator
from typing import Optional
from typing import overload
from typing import Sequence
from typing import Tuple
from typing import Type
from typing import TYPE_CHECKING
from typing import TypeVar
from typing import Union
from .session import _AS
from .session import async_sessionmaker
from .session import AsyncSession
from ... import exc as sa_exc
from ... import util
from ...orm.session import Session
from ...util import create_proxy_methods
from ...util import ScopedRegistry
from ...util import warn
from ...util import warn_deprecated
def expire_all(self) -> None:
    """Expires all persistent instances within this Session.

        .. container:: class_bases

            Proxied for the :class:`_asyncio.AsyncSession` class on
            behalf of the :class:`_asyncio.scoping.async_scoped_session` class.

        .. container:: class_bases

            Proxied for the :class:`_orm.Session` class on
            behalf of the :class:`_asyncio.AsyncSession` class.

        When any attributes on a persistent instance is next accessed,
        a query will be issued using the
        :class:`.Session` object's current transactional context in order to
        load all expired attributes for the given instance.   Note that
        a highly isolated transaction will return the same values as were
        previously read in that same transaction, regardless of changes
        in database state outside of that transaction.

        To expire individual objects and individual attributes
        on those objects, use :meth:`Session.expire`.

        The :class:`.Session` object's default behavior is to
        expire all state whenever the :meth:`Session.rollback`
        or :meth:`Session.commit` methods are called, so that new
        state can be loaded for the new transaction.   For this reason,
        calling :meth:`Session.expire_all` is not usually needed,
        assuming the transaction is isolated.

        .. seealso::

            :ref:`session_expire` - introductory material

            :meth:`.Session.expire`

            :meth:`.Session.refresh`

            :meth:`_orm.Query.populate_existing`



        """
    return self._proxied.expire_all()