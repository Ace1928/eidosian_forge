import functools
import warnings
import json
import contextvars
import flask
from . import exceptions
from ._utils import AttributeDict
@property
@has_context
def args_grouping(self):
    """
        args_grouping is a dict of the inputs used with flexible callback signatures. The keys are the variable names
        and the values are dictionaries containing:
        - “id”: (string or dict) the component id. If it’s a pattern matching id, it will be a dict.
        - “id_str”: (str) for pattern matching ids, it’s the stringified dict id with no white spaces.
        - “property”: (str) The component property used in the callback.
        - “value”: the value of the component property at the time the callback was fired.
        - “triggered”: (bool)Whether this input triggered the callback.

        Example usage:
        @app.callback(
            Output("container", "children"),
            inputs=dict(btn1=Input("btn-1", "n_clicks"), btn2=Input("btn-2", "n_clicks")),
        )
        def display(btn1, btn2):
            c = ctx.args_grouping
            if c.btn1.triggered:
                return f"Button 1 clicked {btn1} times"
            elif c.btn2.triggered:
                return f"Button 2 clicked {btn2} times"
            else:
               return "No clicks yet"

        """
    return getattr(_get_context_value(), 'args_grouping', [])