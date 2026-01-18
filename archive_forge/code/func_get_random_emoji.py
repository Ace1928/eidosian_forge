from __future__ import annotations
import random
from textwrap import dedent
from typing import TYPE_CHECKING, Final, Literal, Mapping, Union, cast
from typing_extensions import TypeAlias
from streamlit.elements import image
from streamlit.errors import StreamlitAPIException
from streamlit.proto.ForwardMsg_pb2 import ForwardMsg as ForwardProto
from streamlit.proto.PageConfig_pb2 import PageConfig as PageConfigProto
from streamlit.runtime.metrics_util import gather_metrics
from streamlit.runtime.scriptrunner import get_script_run_ctx
from streamlit.string_util import is_emoji
from streamlit.url_util import is_url
from streamlit.util import lower_clean_dict_keys
def get_random_emoji() -> str:
    return random.choice(RANDOM_EMOJIS + 10 * ENG_EMOJIS)