from __future__ import annotations
import re
from dataclasses import dataclass
from textwrap import dedent
from typing import cast
import streamlit
from streamlit.elements.form import current_form_id
from streamlit.elements.utils import (
from streamlit.errors import StreamlitAPIException
from streamlit.proto.ColorPicker_pb2 import ColorPicker as ColorPickerProto
from streamlit.runtime.metrics_util import gather_metrics
from streamlit.runtime.scriptrunner import ScriptRunContext, get_script_run_ctx
from streamlit.runtime.state import (
from streamlit.runtime.state.common import compute_widget_id
from streamlit.type_util import Key, LabelVisibility, maybe_raise_label_warnings, to_key
@gather_metrics('color_picker')
def color_picker(self, label: str, value: str | None=None, key: Key | None=None, help: str | None=None, on_change: WidgetCallback | None=None, args: WidgetArgs | None=None, kwargs: WidgetKwargs | None=None, *, disabled: bool=False, label_visibility: LabelVisibility='visible') -> str:
    """Display a color picker widget.

        Parameters
        ----------
        label : str
            A short label explaining to the user what this input is for.
            The label can optionally contain Markdown and supports the following
            elements: Bold, Italics, Strikethroughs, Inline Code, Emojis, and Links.

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

            Unsupported elements are unwrapped so only their children (text contents) render.
            Display unsupported elements as literal characters by
            backslash-escaping them. E.g. ``1\\. Not an ordered list``.

            For accessibility reasons, you should never set an empty label (label="")
            but hide it with label_visibility if needed. In the future, we may disallow
            empty labels by raising an exception.
        value : str
            The hex value of this widget when it first renders. If None,
            defaults to black.
        key : str or int
            An optional string or integer to use as the unique key for the widget.
            If this is omitted, a key will be generated for the widget
            based on its content. Multiple widgets of the same type may
            not share the same key.
        help : str
            An optional tooltip that gets displayed next to the color picker.
        on_change : callable
            An optional callback invoked when this color_picker's value
            changes.
        args : tuple
            An optional tuple of args to pass to the callback.
        kwargs : dict
            An optional dict of kwargs to pass to the callback.
        disabled : bool
            An optional boolean, which disables the color picker if set to
            True. The default is False. This argument can only be supplied by
            keyword.
        label_visibility : "visible", "hidden", or "collapsed"
            The visibility of the label. If "hidden", the label doesn't show but there
            is still empty space for it above the widget (equivalent to label="").
            If "collapsed", both the label and the space are removed. Default is
            "visible".

        Returns
        -------
        str
            The selected color as a hex string.

        Example
        -------
        >>> import streamlit as st
        >>>
        >>> color = st.color_picker('Pick A Color', '#00f900')
        >>> st.write('The current color is', color)

        .. output::
           https://doc-color-picker.streamlit.app/
           height: 335px

        """
    ctx = get_script_run_ctx()
    return self._color_picker(label=label, value=value, key=key, help=help, on_change=on_change, args=args, kwargs=kwargs, disabled=disabled, label_visibility=label_visibility, ctx=ctx)