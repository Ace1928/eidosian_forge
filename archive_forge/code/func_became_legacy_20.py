from __future__ import annotations
import re
from typing import Any
from typing import Callable
from typing import Dict
from typing import Match
from typing import Optional
from typing import Sequence
from typing import Set
from typing import Tuple
from typing import Type
from typing import TypeVar
from typing import Union
from . import compat
from .langhelpers import _hash_limit_string
from .langhelpers import _warnings_warn
from .langhelpers import decorator
from .langhelpers import inject_docstring_text
from .langhelpers import inject_param_text
from .. import exc
def became_legacy_20(api_name: str, alternative: Optional[str]=None, **kw: Any) -> Callable[[_F], _F]:
    type_reg = re.match('^:(attr|func|meth):', api_name)
    if type_reg:
        type_ = {'attr': 'attribute', 'func': 'function', 'meth': 'method'}[type_reg.group(1)]
    else:
        type_ = 'construct'
    message = 'The %s %s is considered legacy as of the 1.x series of SQLAlchemy and %s in 2.0.' % (api_name, type_, 'becomes a legacy construct')
    if ':attr:' in api_name:
        attribute_ok = kw.pop('warn_on_attribute_access', False)
        if not attribute_ok:
            assert kw.get('enable_warnings') is False, 'attribute %s will emit a warning on read access.  If you *really* want this, add warn_on_attribute_access=True.  Otherwise please add enable_warnings=False.' % api_name
    if alternative:
        message += ' ' + alternative
    warning_cls = exc.LegacyAPIWarning
    return deprecated('2.0', message=message, warning=warning_cls, **kw)