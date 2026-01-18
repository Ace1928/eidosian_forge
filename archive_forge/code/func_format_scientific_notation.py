from blib2to3.pytree import Leaf
def format_scientific_notation(text: str) -> str:
    """Formats a numeric string utilizing scientific notation"""
    before, after = text.split('e')
    sign = ''
    if after.startswith('-'):
        after = after[1:]
        sign = '-'
    elif after.startswith('+'):
        after = after[1:]
    before = format_float_or_int_string(before)
    return f'{before}e{sign}{after}'