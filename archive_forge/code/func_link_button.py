from __future__ import annotations
import io
import os
from dataclasses import dataclass
from textwrap import dedent
from typing import TYPE_CHECKING, BinaryIO, Final, Literal, TextIO, Union, cast
from typing_extensions import TypeAlias
from streamlit import runtime, source_util
from streamlit.elements.form import current_form_id, is_in_form
from streamlit.elements.utils import check_callback_rules, check_session_state_rules
from streamlit.errors import StreamlitAPIException
from streamlit.file_util import get_main_script_directory, normalize_path_join
from streamlit.proto.Button_pb2 import Button as ButtonProto
from streamlit.proto.DownloadButton_pb2 import DownloadButton as DownloadButtonProto
from streamlit.proto.LinkButton_pb2 import LinkButton as LinkButtonProto
from streamlit.proto.PageLink_pb2 import PageLink as PageLinkProto
from streamlit.runtime.metrics_util import gather_metrics
from streamlit.runtime.scriptrunner import ScriptRunContext, get_script_run_ctx
from streamlit.runtime.state import (
from streamlit.runtime.state.common import compute_widget_id, save_for_app_testing
from streamlit.string_util import validate_emoji
from streamlit.type_util import Key, to_key
@gather_metrics('link_button')
def link_button(self, label: str, url: str, *, help: str | None=None, type: Literal['primary', 'secondary']='secondary', disabled: bool=False, use_container_width: bool=False) -> DeltaGenerator:
    """Display a link button element.

        When clicked, a new tab will be opened to the specified URL. This will
        create a new session for the user if directed within the app.

        Parameters
        ----------
        label : str
            A short label explaining to the user what this button is for.
            The label can optionally contain Markdown and supports the following
            elements: Bold, Italics, Strikethroughs, Inline Code, and Emojis.

            This also supports:

            * Emoji shortcodes, such as ``:+1:``  and ``:sunglasses:``.
              For a list of all supported codes,
              see https://share.streamlit.io/streamlit/emoji-shortcodes.

            * LaTeX expressions, by wrapping them in "$" or "$$" (the "$$"
              must be on their own lines). Supported LaTeX functions are listed
              at https://katex.org/docs/supported.html.

            * Colored text, using the syntax ``:color[text to be colored]``,
              where ``color`` needs to be replaced with any of the following
              supported colors: blue, green, orange, red, violet, gray/grey, rainbow.

            Unsupported elements are unwrapped so only their children (text contents)
            render. Display unsupported elements as literal characters by
            backslash-escaping them. E.g. ``1\\. Not an ordered list``.
        url : str
            The url to be opened on user click
        help : str
            An optional tooltip that gets displayed when the button is
            hovered over.
        type : "secondary" or "primary"
            An optional string that specifies the button type. Can be "primary" for a
            button with additional emphasis or "secondary" for a normal button. Defaults
            to "secondary".
        disabled : bool
            An optional boolean, which disables the link button if set to
            True. The default is False.
        use_container_width: bool
            An optional boolean, which makes the button stretch its width to match the
            parent container.

        Example
        -------
        >>> import streamlit as st
        >>>
        >>> st.link_button("Go to gallery", "https://streamlit.io/gallery")

        .. output::
           https://doc-link-button.streamlit.app/
           height: 200px

        """
    if type not in ['primary', 'secondary']:
        raise StreamlitAPIException(f'The type argument to st.link_button must be "primary" or "secondary". \nThe argument passed was "{type}".')
    return self._link_button(label=label, url=url, help=help, disabled=disabled, type=type, use_container_width=use_container_width)