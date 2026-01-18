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
def selection_interval(name: Optional[str]=None, value: Union[Any, UndefinedType]=Undefined, bind: Union[core.Binding, str, UndefinedType]=Undefined, empty: Union[bool, UndefinedType]=Undefined, expr: Union[str, core.Expr, _expr_core.Expression, UndefinedType]=Undefined, encodings: Union[List[str], UndefinedType]=Undefined, on: Union[str, UndefinedType]=Undefined, clear: Union[str, bool, UndefinedType]=Undefined, resolve: Union[Literal['global', 'union', 'intersect'], UndefinedType]=Undefined, mark: Union[core.Mark, UndefinedType]=Undefined, translate: Union[str, bool, UndefinedType]=Undefined, zoom: Union[str, bool, UndefinedType]=Undefined, **kwds) -> Parameter:
    """Create an interval selection parameter. Selection parameters define data queries that are driven by direct manipulation from user input (e.g., mouse clicks or drags). Interval selection parameters are used to select a continuous range of data values on drag, whereas point selection parameters (`selection_point`) are used to select multiple discrete data values.)

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
    mark : :class:`Mark` (optional)
        An interval selection also adds a rectangle mark to depict the
        extents of the interval. The mark property can be used to
        customize the appearance of the mark.
    translate : string or boolean (optional)
        When truthy, allows a user to interactively move an interval
        selection back-and-forth. Can be True, False (to disable panning),
        or a Vega event stream definition which must include a start and
        end event to trigger continuous panning. Discrete panning (e.g.,
        pressing the left/right arrow keys) will be supported in future
        versions.
        The default value is True, which corresponds to
        [pointerdown, window:pointerup] > window:pointermove!
        This default allows users to click and drag within an interval
        selection to reposition it.
    zoom : string or boolean (optional)
        When truthy, allows a user to interactively resize an interval
        selection. Can be True, False (to disable zooming), or a Vega
        event stream definition. Currently, only wheel events are supported,
        but custom event streams can still be used to specify filters,
        debouncing, and throttling. Future versions will expand the set of
        events that can trigger this transformation.
        The default value is True, which corresponds to wheel!. This
        default allows users to use the mouse wheel to resize an interval
        selection.
    **kwds :
        Additional keywords to control the selection.

    Returns
    -------
    parameter: Parameter
        The parameter object that can be used in chart creation.
    """
    return _selection(type='interval', name=name, value=value, bind=bind, empty=empty, expr=expr, encodings=encodings, on=on, clear=clear, resolve=resolve, mark=mark, translate=translate, zoom=zoom, **kwds)