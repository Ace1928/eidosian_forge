from __future__ import annotations
from typing import Any
from typing import Callable
from typing import Collection
from typing import ContextManager
from typing import Dict
from typing import List
from typing import Mapping
from typing import MutableMapping
from typing import Optional
from typing import overload
from typing import Sequence
from typing import TextIO
from typing import Tuple
from typing import TYPE_CHECKING
from typing import Union
from sqlalchemy.sql.schema import Column
from sqlalchemy.sql.schema import FetchedValue
from typing_extensions import Literal
from .migration import _ProxyTransaction
from .migration import MigrationContext
from .. import util
from ..operations import Operations
from ..script.revision import _GetRevArg
def get_x_argument(self, as_dictionary: bool=False) -> Union[List[str], Dict[str, str]]:
    """Return the value(s) passed for the ``-x`` argument, if any.

        The ``-x`` argument is an open ended flag that allows any user-defined
        value or values to be passed on the command line, then available
        here for consumption by a custom ``env.py`` script.

        The return value is a list, returned directly from the ``argparse``
        structure.  If ``as_dictionary=True`` is passed, the ``x`` arguments
        are parsed using ``key=value`` format into a dictionary that is
        then returned. If there is no ``=`` in the argument, value is an empty
        string.

        .. versionchanged:: 1.13.1 Support ``as_dictionary=True`` when
           arguments are passed without the ``=`` symbol.

        For example, to support passing a database URL on the command line,
        the standard ``env.py`` script can be modified like this::

            cmd_line_url = context.get_x_argument(
                as_dictionary=True).get('dbname')
            if cmd_line_url:
                engine = create_engine(cmd_line_url)
            else:
                engine = engine_from_config(
                        config.get_section(config.config_ini_section),
                        prefix='sqlalchemy.',
                        poolclass=pool.NullPool)

        This then takes effect by running the ``alembic`` script as::

            alembic -x dbname=postgresql://user:pass@host/dbname upgrade head

        This function does not require that the :class:`.MigrationContext`
        has been configured.

        .. seealso::

            :meth:`.EnvironmentContext.get_tag_argument`

            :attr:`.Config.cmd_opts`

        """
    if self.config.cmd_opts is not None:
        value = self.config.cmd_opts.x or []
    else:
        value = []
    if as_dictionary:
        dict_value = {}
        for arg in value:
            x_key, _, x_value = arg.partition('=')
            dict_value[x_key] = x_value
        value = dict_value
    return value