import warnings
import hashlib
import io
import json
import jsonschema
import pandas as pd
from toolz.curried import pipe as _pipe
import itertools
import sys
from typing import cast, List, Optional, Any, Iterable, Union, Literal, IO
from typing import Type as TypingType
from typing import Dict as TypingDict
from .schema import core, channels, mixins, Undefined, UndefinedType, SCHEMA_URL
from .data import data_transformers
from ... import utils, expr
from ...expr import core as _expr_core
from .display import renderers, VEGALITE_VERSION, VEGAEMBED_VERSION, VEGA_VERSION
from .theme import themes
from .compiler import vegalite_compilers
from ...utils._vegafusion_data import (
from ...utils.core import DataFrameLike
from ...utils.data import DataType
def selection_point(name: Optional[str]=None, value: Union[Any, UndefinedType]=Undefined, bind: Union[core.Binding, str, UndefinedType]=Undefined, empty: Union[bool, UndefinedType]=Undefined, expr: Union[core.Expr, UndefinedType]=Undefined, encodings: Union[List[str], UndefinedType]=Undefined, fields: Union[List[str], UndefinedType]=Undefined, on: Union[str, UndefinedType]=Undefined, clear: Union[str, bool, UndefinedType]=Undefined, resolve: Union[Literal['global', 'union', 'intersect'], UndefinedType]=Undefined, toggle: Union[str, bool, UndefinedType]=Undefined, nearest: Union[bool, UndefinedType]=Undefined, **kwds) -> Parameter:
    """Create a point selection parameter. Selection parameters define data queries that are driven by direct manipulation from user input (e.g., mouse clicks or drags). Point selection parameters are used to select multiple discrete data values; the first value is selected on click and additional values toggled on shift-click. To select a continuous range of data values on drag interval selection parameters (`selection_interval`) can be used instead.

    Parameters
    ----------
    name : string (optional)
        The name of the parameter. If not specified, a unique name will be
        created.
    value : any (optional)
        The default value of the parameter. If not specified, the parameter
        will be created without a default value.
    bind : :class:`Binding`, str (optional)
        Binds the parameter to an external input element such as a slider,
        selection list or radio button group.
    empty : boolean (optional)
        For selection parameters, the predicate of empty selections returns
        True by default. Override this behavior, by setting this property
        'empty=False'.
    expr : :class:`Expr` (optional)
        An expression for the value of the parameter. This expression may
        include other parameters, in which case the parameter will
        automatically update in response to upstream parameter changes.
    encodings : List[str] (optional)
        A list of encoding channels. The corresponding data field values
        must match for a data tuple to fall within the selection.
    fields : List[str] (optional)
        A list of field names whose values must match for a data tuple to
        fall within the selection.
    on : string (optional)
        A Vega event stream (object or selector) that triggers the selection.
        For interval selections, the event stream must specify a start and end.
    clear : string or boolean (optional)
        Clears the selection, emptying it of all values. This property can
        be an Event Stream or False to disable clear.  Default is 'dblclick'.
    resolve : enum('global', 'union', 'intersect') (optional)
        With layered and multi-view displays, a strategy that determines
        how selections' data queries are resolved when applied in a filter
        transform, conditional encoding rule, or scale domain.
        One of:

        * 'global': only one brush exists for the entire SPLOM. When the
          user begins to drag, any previous brushes are cleared, and a
          new one is constructed.
        * 'union': each cell contains its own brush, and points are
          highlighted if they lie within any of these individual brushes.
        * 'intersect': each cell contains its own brush, and points are
          highlighted only if they fall within all of these individual
          brushes.

        The default is 'global'.
    toggle : string or boolean (optional)
        Controls whether data values should be toggled (inserted or
        removed from a point selection) or only ever inserted into
        point selections.
        One of:

        * True (default): the toggle behavior, which corresponds to
          "event.shiftKey". As a result, data values are toggled
          when the user interacts with the shift-key pressed.
        * False: disables toggling behaviour; the selection will
          only ever contain a single data value corresponding
          to the most recent interaction.
        * A Vega expression which is re-evaluated as the user interacts.
          If the expression evaluates to True, the data value is
          toggled into or out of the point selection. If the expression
          evaluates to False, the point selection is first cleared, and
          the data value is then inserted. For example, setting the
          value to the Vega expression True will toggle data values
          without the user pressing the shift-key.

    nearest : boolean (optional)
        When true, an invisible voronoi diagram is computed to accelerate
        discrete selection. The data value nearest the mouse cursor is
        added to the selection.  The default is False, which means that
        data values must be interacted with directly (e.g., clicked on)
        to be added to the selection.
    **kwds :
        Additional keywords to control the selection.

    Returns
    -------
    parameter: Parameter
        The parameter object that can be used in chart creation.
    """
    return _selection(type='point', name=name, value=value, bind=bind, empty=empty, expr=expr, encodings=encodings, fields=fields, on=on, clear=clear, resolve=resolve, toggle=toggle, nearest=nearest, **kwds)