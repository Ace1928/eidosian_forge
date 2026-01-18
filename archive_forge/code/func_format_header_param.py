from __future__ import annotations
import email.utils
import mimetypes
import typing
def format_header_param(name: str, value: _TYPE_FIELD_VALUE) -> str:
    """
    .. deprecated:: 2.0.0
        Renamed to :func:`format_multipart_header_param`. Will be
        removed in urllib3 v2.1.0.
    """
    import warnings
    warnings.warn("'format_header_param' has been renamed to 'format_multipart_header_param'. The old name will be removed in urllib3 v2.1.0.", DeprecationWarning, stacklevel=2)
    return format_multipart_header_param(name, value)