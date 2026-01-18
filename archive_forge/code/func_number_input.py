from __future__ import annotations
import numbers
from dataclasses import dataclass
from textwrap import dedent
from typing import TYPE_CHECKING, Literal, Union, cast, overload
from typing_extensions import TypeAlias
from streamlit.elements.form import current_form_id
from streamlit.elements.utils import (
from streamlit.errors import StreamlitAPIException
from streamlit.js_number import JSNumber, JSNumberBoundsException
from streamlit.proto.NumberInput_pb2 import NumberInput as NumberInputProto
from streamlit.runtime.metrics_util import gather_metrics
from streamlit.runtime.scriptrunner import ScriptRunContext, get_script_run_ctx
from streamlit.runtime.state import (
from streamlit.runtime.state.common import compute_widget_id
from streamlit.type_util import Key, LabelVisibility, maybe_raise_label_warnings, to_key
@gather_metrics('number_input')
def number_input(self, label: str, min_value: Number | None=None, max_value: Number | None=None, value: Number | Literal['min'] | None='min', step: Number | None=None, format: str | None=None, key: Key | None=None, help: str | None=None, on_change: WidgetCallback | None=None, args: WidgetArgs | None=None, kwargs: WidgetKwargs | None=None, *, placeholder: str | None=None, disabled: bool=False, label_visibility: LabelVisibility='visible') -> Number | None:
    """Display a numeric input widget.

        .. note::
            Integer values exceeding +/- ``(1<<53) - 1`` cannot be accurately
            stored or returned by the widget due to serialization contstraints
            between the Python server and JavaScript client. You must handle
            such numbers as floats, leading to a loss in precision.

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
        min_value : int, float, or None
            The minimum permitted value.
            If None, there will be no minimum.
        max_value : int, float, or None
            The maximum permitted value.
            If None, there will be no maximum.
        value : int, float, "min" or None
            The value of this widget when it first renders. If ``None``, will initialize
            empty and return ``None`` until the user provides input.
            If "min" (default), will initialize with min_value, or 0.0 if
            min_value is None.
        step : int, float, or None
            The stepping interval.
            Defaults to 1 if the value is an int, 0.01 otherwise.
            If the value is not specified, the format parameter will be used.
        format : str or None
            A printf-style format string controlling how the interface should
            display numbers. Output must be purely numeric. This does not impact
            the return value. Valid formatters: %d %e %f %g %i %u
        key : str or int
            An optional string or integer to use as the unique key for the widget.
            If this is omitted, a key will be generated for the widget
            based on its content. Multiple widgets of the same type may
            not share the same key.
        help : str
            An optional tooltip that gets displayed next to the input.
        on_change : callable
            An optional callback invoked when this number_input's value changes.
        args : tuple
            An optional tuple of args to pass to the callback.
        kwargs : dict
            An optional dict of kwargs to pass to the callback.
        placeholder : str or None
            An optional string displayed when the number input is empty.
            If None, no placeholder is displayed.
        disabled : bool
            An optional boolean, which disables the number input if set to
            True. The default is False.
        label_visibility : "visible", "hidden", or "collapsed"
            The visibility of the label. If "hidden", the label doesn't show but there
            is still empty space for it above the widget (equivalent to label="").
            If "collapsed", both the label and the space are removed. Default is
            "visible".

        Returns
        -------
        int or float or None
            The current value of the numeric input widget or ``None`` if the widget
            is empty. The return type will match the data type of the value parameter.

        Example
        -------
        >>> import streamlit as st
        >>>
        >>> number = st.number_input('Insert a number')
        >>> st.write('The current number is ', number)

        .. output::
           https://doc-number-input.streamlit.app/
           height: 260px

        To initialize an empty number input, use ``None`` as the value:

        >>> import streamlit as st
        >>>
        >>> number = st.number_input("Insert a number", value=None, placeholder="Type a number...")
        >>> st.write('The current number is ', number)

        .. output::
           https://doc-number-input-empty.streamlit.app/
           height: 260px

        """
    ctx = get_script_run_ctx()
    return self._number_input(label=label, min_value=min_value, max_value=max_value, value=value, step=step, format=format, key=key, help=help, on_change=on_change, args=args, kwargs=kwargs, placeholder=placeholder, disabled=disabled, label_visibility=label_visibility, ctx=ctx)