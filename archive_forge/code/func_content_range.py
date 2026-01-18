from __future__ import annotations
import typing as t
from datetime import datetime
from datetime import timedelta
from datetime import timezone
from http import HTTPStatus
from ..datastructures import Headers
from ..datastructures import HeaderSet
from ..http import dump_cookie
from ..http import HTTP_STATUS_CODES
from ..utils import get_content_type
from werkzeug.datastructures import CallbackDict
from werkzeug.datastructures import ContentRange
from werkzeug.datastructures import ContentSecurityPolicy
from werkzeug.datastructures import ResponseCacheControl
from werkzeug.datastructures import WWWAuthenticate
from werkzeug.http import COEP
from werkzeug.http import COOP
from werkzeug.http import dump_age
from werkzeug.http import dump_header
from werkzeug.http import dump_options_header
from werkzeug.http import http_date
from werkzeug.http import parse_age
from werkzeug.http import parse_cache_control_header
from werkzeug.http import parse_content_range_header
from werkzeug.http import parse_csp_header
from werkzeug.http import parse_date
from werkzeug.http import parse_options_header
from werkzeug.http import parse_set_header
from werkzeug.http import quote_etag
from werkzeug.http import unquote_etag
from werkzeug.utils import header_property
@content_range.setter
def content_range(self, value: ContentRange | str | None) -> None:
    if not value:
        del self.headers['content-range']
    elif isinstance(value, str):
        self.headers['Content-Range'] = value
    else:
        self.headers['Content-Range'] = value.to_header()