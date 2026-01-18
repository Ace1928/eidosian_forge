from blib2to3.pytree import Leaf
def format_hex(text: str) -> str:
    """
    Formats a hexadecimal string like "0x12B3"
    """
    before, after = (text[:2], text[2:])
    return f'{before}{after.upper()}'