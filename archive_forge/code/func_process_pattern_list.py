imported when doing ``import xarray_einstats``.
import warnings
from collections.abc import Hashable
import einops
import xarray as xr
def process_pattern_list(redims, handler, allow_dict=True, allow_list=True):
    """Process a pattern list and convert it to an einops expression using placeholders.

    Parameters
    ----------
    redims : pattern_list
        One of ``out_dims`` or ``in_dims`` in {func}`~xarray_einstats.einops.rearrange`
        or {func}`~xarray_einstats.einops.reduce`.
    handler : DimHandler
    allow_dict, allow_list : bool, optional
        Whether or not to allow lists or dicts as elements of ``redims``.
        When processing ``in_dims`` for example we need the names of
        the variables to be decomposed so dicts are required and lists
        are not accepted.

    Returns
    -------
    expression_dims : list of str
        A list with the names of the dimensions present in the out expression
    output_dims : list of str
        A list with the names of the dimensions present in the output.
        It differs from ``expression_dims`` because there might be dimensions
        being stacked.
    pattern : str
        The einops expression equivalent to the operations in ``redims`` pattern
        list.

    Examples
    --------
    Whenever we have groupings of dimensions (be it to decompose or to stack),
    ``expression_dims`` and ``output_dims`` differ:

    .. jupyter-execute::

        from xarray_einstats.einops import process_pattern_list, DimHandler
        handler = DimHandler()
        process_pattern_list(["a", {"b": ["c", "d"]}, ["e", "f", "g"]], handler)

    """
    out = []
    out_names = []
    txt = []
    for subitem in redims:
        if isinstance(subitem, Hashable):
            out.append(subitem)
            out_names.append(subitem)
            txt.append(handler.get_name(subitem))
        elif isinstance(subitem, dict) and allow_dict:
            if len(subitem) != 1:
                raise ValueError(f'dicts in pattern list must have a single key but instead found {len(subitem)}: {subitem.keys()}')
            key, values = list(subitem.items())[0]
            if isinstance(values, Hashable):
                raise ValueError('Found values of hashable type in a pattern dict, use xarray.rename')
            out.extend(values)
            out_names.append(key)
            txt.append(f'( {handler.get_names(values)} )')
        elif allow_list:
            out.extend(subitem)
            out_names.append('-'.join(subitem))
            txt.append(f'( {handler.get_names(subitem)} )')
        else:
            raise ValueError(f'Found unsupported pattern type: {type(subitem)}, double check the docs. This could be for example is using lists/tuples as elements of in_dims argument')
    return (out, out_names, ' '.join(txt))