from blib2to3.pytree import Leaf
def format_complex_number(text: str) -> str:
    """Formats a complex string like `10j`"""
    number = text[:-1]
    suffix = text[-1]
    return f'{format_float_or_int_string(number)}{suffix}'