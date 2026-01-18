from __future__ import annotations
def make_flex_doc(op_name: str, typ: str) -> str:
    """
    Make the appropriate substitutions for the given operation and class-typ
    into either _flex_doc_SERIES or _flex_doc_FRAME to return the docstring
    to attach to a generated method.

    Parameters
    ----------
    op_name : str {'__add__', '__sub__', ... '__eq__', '__ne__', ...}
    typ : str {series, 'dataframe']}

    Returns
    -------
    doc : str
    """
    op_name = op_name.replace('__', '')
    op_desc = _op_descriptions[op_name]
    op_desc_op = op_desc['op']
    assert op_desc_op is not None
    if op_name.startswith('r'):
        equiv = f'other {op_desc_op} {typ}'
    elif op_name == 'divmod':
        equiv = f'{op_name}({typ}, other)'
    else:
        equiv = f'{typ} {op_desc_op} other'
    if typ == 'series':
        base_doc = _flex_doc_SERIES
        if op_desc['reverse']:
            base_doc += _see_also_reverse_SERIES.format(reverse=op_desc['reverse'], see_also_desc=op_desc['see_also_desc'])
        doc_no_examples = base_doc.format(desc=op_desc['desc'], op_name=op_name, equiv=equiv, series_returns=op_desc['series_returns'])
        ser_example = op_desc['series_examples']
        if ser_example:
            doc = doc_no_examples + ser_example
        else:
            doc = doc_no_examples
    elif typ == 'dataframe':
        if op_name in ['eq', 'ne', 'le', 'lt', 'ge', 'gt']:
            base_doc = _flex_comp_doc_FRAME
            doc = _flex_comp_doc_FRAME.format(op_name=op_name, desc=op_desc['desc'])
        else:
            base_doc = _flex_doc_FRAME
            doc = base_doc.format(desc=op_desc['desc'], op_name=op_name, equiv=equiv, reverse=op_desc['reverse'])
    else:
        raise AssertionError('Invalid typ argument.')
    return doc