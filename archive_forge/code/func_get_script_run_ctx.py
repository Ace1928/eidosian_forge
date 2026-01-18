from __future__ import annotations
import collections
import threading
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Callable, Counter, Dict, Final, Union
from urllib import parse
from typing_extensions import TypeAlias
from streamlit import runtime
from streamlit.errors import StreamlitAPIException
from streamlit.logger import get_logger
from streamlit.proto.ForwardMsg_pb2 import ForwardMsg
from streamlit.proto.PageProfile_pb2 import Command
from streamlit.runtime.scriptrunner.script_requests import ScriptRequests
from streamlit.runtime.state import SafeSessionState
from streamlit.runtime.uploaded_file_manager import UploadedFileManager
import streamlit
def get_script_run_ctx(suppress_warning: bool=False) -> ScriptRunContext | None:
    """
    Parameters
    ----------
    suppress_warning : bool
        If True, don't log a warning if there's no ScriptRunContext.
    Returns
    -------
    ScriptRunContext | None
        The current thread's ScriptRunContext, or None if it doesn't have one.

    """
    thread = threading.current_thread()
    ctx: ScriptRunContext | None = getattr(thread, SCRIPT_RUN_CONTEXT_ATTR_NAME, None)
    if ctx is None and runtime.exists() and (not suppress_warning):
        _LOGGER.warning("Thread '%s': missing ScriptRunContext", thread.name)
    return ctx