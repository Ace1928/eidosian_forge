from __future__ import annotations
import re
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta
from textwrap import dedent
from typing import (
from typing_extensions import TypeAlias
from streamlit.elements.form import current_form_id
from streamlit.elements.utils import (
from streamlit.errors import StreamlitAPIException
from streamlit.proto.DateInput_pb2 import DateInput as DateInputProto
from streamlit.proto.TimeInput_pb2 import TimeInput as TimeInputProto
from streamlit.runtime.metrics_util import gather_metrics
from streamlit.runtime.scriptrunner import ScriptRunContext, get_script_run_ctx
from streamlit.runtime.state import (
from streamlit.runtime.state.common import compute_widget_id
from streamlit.time_util import adjust_years
from streamlit.type_util import Key, LabelVisibility, maybe_raise_label_warnings, to_key
@gather_metrics('date_input')
def date_input(self, label: str, value: DateValue | Literal['today']='default_value_today', min_value: SingleDateValue=None, max_value: SingleDateValue=None, key: Key | None=None, help: str | None=None, on_change: WidgetCallback | None=None, args: WidgetArgs | None=None, kwargs: WidgetKwargs | None=None, *, format: str='YYYY/MM/DD', disabled: bool=False, label_visibility: LabelVisibility='visible') -> DateWidgetReturn:
    """Display a date input widget.

        Parameters
        ----------
        label : str
            A short label explaining to the user what this date input is for.
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
        value : datetime.date or datetime.datetime or list/tuple of datetime.date or datetime.datetime, "today", or None
            The value of this widget when it first renders. If a list/tuple with
            0 to 2 date/datetime values is provided, the datepicker will allow
            users to provide a range. If ``None``, will initialize empty and
            return ``None`` until the user provides input. If "today" (default),
            will initialize with today as a single-date picker.
        min_value : datetime.date or datetime.datetime
            The minimum selectable date. If value is a date, defaults to value - 10 years.
            If value is the interval [start, end], defaults to start - 10 years.
        max_value : datetime.date or datetime.datetime
            The maximum selectable date. If value is a date, defaults to value + 10 years.
            If value is the interval [start, end], defaults to end + 10 years.
        key : str or int
            An optional string or integer to use as the unique key for the widget.
            If this is omitted, a key will be generated for the widget
            based on its content. Multiple widgets of the same type may
            not share the same key.
        help : str
            An optional tooltip that gets displayed next to the input.
        on_change : callable
            An optional callback invoked when this date_input's value changes.
        args : tuple
            An optional tuple of args to pass to the callback.
        kwargs : dict
            An optional dict of kwargs to pass to the callback.
        format : str
            A format string controlling how the interface should display dates.
            Supports “YYYY/MM/DD” (default), “DD/MM/YYYY”, or “MM/DD/YYYY”.
            You may also use a period (.) or hyphen (-) as separators.
        disabled : bool
            An optional boolean, which disables the date input if set to True.
            The default is False.
        label_visibility : "visible", "hidden", or "collapsed"
            The visibility of the label. If "hidden", the label doesn't show but there
            is still empty space for it above the widget (equivalent to label="").
            If "collapsed", both the label and the space are removed. Default is
            "visible".


        Returns
        -------
        datetime.date or a tuple with 0-2 dates or None
            The current value of the date input widget or ``None`` if no date has been
            selected.

        Examples
        --------
        >>> import datetime
        >>> import streamlit as st
        >>>
        >>> d = st.date_input("When's your birthday", datetime.date(2019, 7, 6))
        >>> st.write('Your birthday is:', d)

        .. output::
           https://doc-date-input.streamlit.app/
           height: 380px

        >>> import datetime
        >>> import streamlit as st
        >>>
        >>> today = datetime.datetime.now()
        >>> next_year = today.year + 1
        >>> jan_1 = datetime.date(next_year, 1, 1)
        >>> dec_31 = datetime.date(next_year, 12, 31)
        >>>
        >>> d = st.date_input(
        ...     "Select your vacation for next year",
        ...     (jan_1, datetime.date(next_year, 1, 7)),
        ...     jan_1,
        ...     dec_31,
        ...     format="MM.DD.YYYY",
        ... )
        >>> d

        .. output::
           https://doc-date-input1.streamlit.app/
           height: 380px

        To initialize an empty date input, use ``None`` as the value:

        >>> import datetime
        >>> import streamlit as st
        >>>
        >>> d = st.date_input("When's your birthday", value=None)
        >>> st.write('Your birthday is:', d)

        .. output::
           https://doc-date-input-empty.streamlit.app/
           height: 380px

        """
    ctx = get_script_run_ctx()
    return self._date_input(label=label, value=value, min_value=min_value, max_value=max_value, key=key, help=help, on_change=on_change, args=args, kwargs=kwargs, disabled=disabled, label_visibility=label_visibility, format=format, ctx=ctx)