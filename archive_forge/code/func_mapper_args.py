from __future__ import annotations
import collections
import contextlib
from typing import Any
from typing import Callable
from typing import TYPE_CHECKING
from typing import Union
from ... import exc as sa_exc
from ...engine import Connection
from ...engine import Engine
from ...orm import exc as orm_exc
from ...orm import relationships
from ...orm.base import _mapper_or_none
from ...orm.clsregistry import _resolver
from ...orm.decl_base import _DeferredMapperConfig
from ...orm.util import polymorphic_union
from ...schema import Table
from ...util import OrderedDict
def mapper_args():
    args = m_args()
    args['polymorphic_on'] = pjoin.c[discriminator_name]
    args['polymorphic_abstract'] = True
    if strict_attrs:
        args['include_properties'] = set(pjoin.primary_key) | declared_col_keys | {discriminator_name}
        args['with_polymorphic'] = ('*', pjoin)
    return args