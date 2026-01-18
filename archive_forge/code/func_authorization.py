from __future__ import annotations
import typing as t
from datetime import datetime
from urllib.parse import parse_qsl
from ..datastructures import Accept
from ..datastructures import Authorization
from ..datastructures import CharsetAccept
from ..datastructures import ETags
from ..datastructures import Headers
from ..datastructures import HeaderSet
from ..datastructures import IfRange
from ..datastructures import ImmutableList
from ..datastructures import ImmutableMultiDict
from ..datastructures import LanguageAccept
from ..datastructures import MIMEAccept
from ..datastructures import MultiDict
from ..datastructures import Range
from ..datastructures import RequestCacheControl
from ..http import parse_accept_header
from ..http import parse_cache_control_header
from ..http import parse_date
from ..http import parse_etags
from ..http import parse_if_range_header
from ..http import parse_list_header
from ..http import parse_options_header
from ..http import parse_range_header
from ..http import parse_set_header
from ..user_agent import UserAgent
from ..utils import cached_property
from ..utils import header_property
from .http import parse_cookie
from .utils import get_content_length
from .utils import get_current_url
from .utils import get_host
@cached_property
def authorization(self) -> Authorization | None:
    """The ``Authorization`` header parsed into an :class:`.Authorization` object.
        ``None`` if the header is not present.

        .. versionchanged:: 2.3
            :class:`Authorization` is no longer a ``dict``. The ``token`` attribute
            was added for auth schemes that use a token instead of parameters.
        """
    return Authorization.from_header(self.headers.get('Authorization'))