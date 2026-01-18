import contextlib
import os
import re
import sys
import sentry_sdk
from sentry_sdk.consts import OP, SPANDATA
from sentry_sdk.utils import (
from sentry_sdk._compat import PY2, duration_in_milliseconds, iteritems
from sentry_sdk._types import TYPE_CHECKING
from sentry_sdk.tracing import LOW_QUALITY_TRANSACTION_SOURCES
def add_query_source(hub, span):
    """
    Adds OTel compatible source code information to the span
    """
    client = hub.client
    if client is None:
        return
    if span.timestamp is None or span.start_timestamp is None:
        return
    should_add_query_source = client.options.get('enable_db_query_source', True)
    if not should_add_query_source:
        return
    duration = span.timestamp - span.start_timestamp
    threshold = client.options.get('db_query_source_threshold_ms', 0)
    slow_query = duration_in_milliseconds(duration) > threshold
    if not slow_query:
        return
    project_root = client.options['project_root']
    in_app_include = client.options.get('in_app_include')
    in_app_exclude = client.options.get('in_app_exclude')
    frame = sys._getframe()
    while frame is not None:
        try:
            abs_path = frame.f_code.co_filename
            if abs_path and PY2:
                abs_path = os.path.abspath(abs_path)
        except Exception:
            abs_path = ''
        try:
            namespace = frame.f_globals.get('__name__')
        except Exception:
            namespace = None
        is_sentry_sdk_frame = namespace is not None and namespace.startswith('sentry_sdk.')
        should_be_included = not _is_external_source(abs_path)
        if namespace is not None:
            if in_app_exclude and _module_in_list(namespace, in_app_exclude):
                should_be_included = False
            if in_app_include and _module_in_list(namespace, in_app_include):
                should_be_included = True
        if abs_path.startswith(project_root) and should_be_included and (not is_sentry_sdk_frame):
            break
        frame = frame.f_back
    else:
        frame = None
    if frame is not None:
        try:
            lineno = frame.f_lineno
        except Exception:
            lineno = None
        if lineno is not None:
            span.set_data(SPANDATA.CODE_LINENO, frame.f_lineno)
        try:
            namespace = frame.f_globals.get('__name__')
        except Exception:
            namespace = None
        if namespace is not None:
            span.set_data(SPANDATA.CODE_NAMESPACE, namespace)
        try:
            filepath = frame.f_code.co_filename
        except Exception:
            filepath = None
        if filepath is not None:
            if namespace is not None and (not PY2):
                in_app_path = filename_for_module(namespace, filepath)
            elif project_root is not None and filepath.startswith(project_root):
                in_app_path = filepath.replace(project_root, '').lstrip(os.sep)
            else:
                in_app_path = filepath
            span.set_data(SPANDATA.CODE_FILEPATH, in_app_path)
        try:
            code_function = frame.f_code.co_name
        except Exception:
            code_function = None
        if code_function is not None:
            span.set_data(SPANDATA.CODE_FUNCTION, frame.f_code.co_name)