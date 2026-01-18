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
@property
def no_autoflush(self) -> Any:
    """Return a context manager that disables autoflush.

        .. container:: class_bases

            Proxied for the :class:`_asyncio.AsyncSession` class
            on behalf of the :class:`_asyncio.scoping.async_scoped_session` class.

        .. container:: class_bases

            Proxied for the :class:`_orm.Session` class
            on behalf of the :class:`_asyncio.AsyncSession` class.

        e.g.::

            with session.no_autoflush:

                some_object = SomeClass()
                session.add(some_object)
                # won't autoflush
                some_object.related_thing = session.query(SomeRelated).first()

        Operations that proceed within the ``with:`` block
        will not be subject to flushes occurring upon query
        access.  This is useful when initializing a series
        of objects which involve existing database queries,
        where the uncompleted object should not yet be flushed.



        """
    return self._proxied.no_autoflush