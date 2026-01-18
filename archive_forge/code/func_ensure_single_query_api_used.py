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
def ensure_single_query_api_used(self):
    if self._experimental_query_params_used and self._production_query_params_used:
        raise StreamlitAPIException('Using `st.query_params` together with either `st.experimental_get_query_params` ' + 'or `st.experimental_set_query_params` is not supported. Please convert your app ' + 'to only use `st.query_params`')