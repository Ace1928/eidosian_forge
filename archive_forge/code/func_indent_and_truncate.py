import sys
from osc_lib.command import command
def indent_and_truncate(txt, spaces=0, truncate=False, truncate_limit=10, truncate_prefix=None, truncate_postfix=None):
    """Indents supplied multiline text by the specified number of spaces

    """
    if txt is None:
        return
    lines = str(txt).splitlines()
    if truncate and len(lines) > truncate_limit:
        lines = lines[-truncate_limit:]
        if truncate_prefix is not None:
            lines.insert(0, truncate_prefix)
        if truncate_postfix is not None:
            lines.append(truncate_postfix)
    if spaces > 0:
        lines = [' ' * spaces + line for line in lines]
    return '\n'.join(lines)