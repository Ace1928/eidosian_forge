from __future__ import annotations
from collections import defaultdict
from collections.abc import Hashable, Iterable, Mapping, MutableMapping
from typing import TYPE_CHECKING, Any, Literal, Union
import numpy as np
import pandas as pd
from xarray.coding import strings, times, variables
from xarray.coding.variables import SerializationWarning, pop_to
from xarray.core import indexing
from xarray.core.common import (
from xarray.core.utils import emit_user_level_warning
from xarray.core.variable import IndexVariable, Variable
from xarray.namedarray.utils import is_duck_dask_array
def decode_cf_variable(name: Hashable, var: Variable, concat_characters: bool=True, mask_and_scale: bool=True, decode_times: bool=True, decode_endianness: bool=True, stack_char_dim: bool=True, use_cftime: bool | None=None, decode_timedelta: bool | None=None) -> Variable:
    """
    Decodes a variable which may hold CF encoded information.

    This includes variables that have been masked and scaled, which
    hold CF style time variables (this is almost always the case if
    the dataset has been serialized) and which have strings encoded
    as character arrays.

    Parameters
    ----------
    name : str
        Name of the variable. Used for better error messages.
    var : Variable
        A variable holding potentially CF encoded information.
    concat_characters : bool
        Should character arrays be concatenated to strings, for
        example: ["h", "e", "l", "l", "o"] -> "hello"
    mask_and_scale : bool
        Lazily scale (using scale_factor and add_offset) and mask
        (using _FillValue). If the _Unsigned attribute is present
        treat integer arrays as unsigned.
    decode_times : bool
        Decode cf times ("hours since 2000-01-01") to np.datetime64.
    decode_endianness : bool
        Decode arrays from non-native to native endianness.
    stack_char_dim : bool
        Whether to stack characters into bytes along the last dimension of this
        array. Passed as an argument because we need to look at the full
        dataset to figure out if this is appropriate.
    use_cftime : bool, optional
        Only relevant if encoded dates come from a standard calendar
        (e.g. "gregorian", "proleptic_gregorian", "standard", or not
        specified).  If None (default), attempt to decode times to
        ``np.datetime64[ns]`` objects; if this is not possible, decode times to
        ``cftime.datetime`` objects. If True, always decode times to
        ``cftime.datetime`` objects, regardless of whether or not they can be
        represented using ``np.datetime64[ns]`` objects.  If False, always
        decode times to ``np.datetime64[ns]`` objects; if this is not possible
        raise an error.

    Returns
    -------
    out : Variable
        A variable holding the decoded equivalent of var.
    """
    if _contains_datetime_like_objects(var):
        return var
    original_dtype = var.dtype
    if decode_timedelta is None:
        decode_timedelta = decode_times
    if concat_characters:
        if stack_char_dim:
            var = strings.CharacterArrayCoder().decode(var, name=name)
        var = strings.EncodedStringCoder().decode(var)
    if original_dtype == object:
        var = variables.ObjectVLenStringCoder().decode(var)
        original_dtype = var.dtype
    if mask_and_scale:
        for coder in [variables.UnsignedIntegerCoder(), variables.CFMaskCoder(), variables.CFScaleOffsetCoder()]:
            var = coder.decode(var, name=name)
    if decode_timedelta:
        var = times.CFTimedeltaCoder().decode(var, name=name)
    if decode_times:
        var = times.CFDatetimeCoder(use_cftime=use_cftime).decode(var, name=name)
    if decode_endianness and (not var.dtype.isnative):
        var = variables.EndianCoder().decode(var)
        original_dtype = var.dtype
    var = variables.BooleanCoder().decode(var)
    dimensions, data, attributes, encoding = variables.unpack_for_decoding(var)
    encoding.setdefault('dtype', original_dtype)
    if not is_duck_dask_array(data):
        data = indexing.LazilyIndexedArray(data)
    return Variable(dimensions, data, attributes, encoding=encoding, fastpath=True)