import os
import os.path
import re
from debian.deprecation import function_deprecated_by
import debian._arch_table
def patches_from_ed_script(source, re_cmd=None):
    """Converts source to a stream of patches.

    Patches are triples of line indexes:

    - number of the first line to be replaced
    - one plus the number of the last line to be replaced
    - list of line replacements

    This is enough to model arbitrary additions, deletions and
    replacements.
    """
    i = iter(source)
    patch_re = re_cmd
    for line in i:
        if not patch_re:
            patch_re = _patch_re_b if isinstance(line, bytes) else _patch_re
        match = patch_re.match(line)
        if match is None:
            raise ValueError('invalid patch command: %r' % line)
        first_, last_, cmd = match.groups()
        first = int(first_)
        last = None if last_ is None else int(last_)
        if ord(cmd) == 100:
            first = first - 1
            if last is None:
                last = first + 1
            yield (first, last, [])
            continue
        if ord(cmd) == 97:
            if last is not None:
                raise ValueError('invalid patch argument: %r' % line)
            last = first
        else:
            first = first - 1
            if last is None:
                last = first + 1
        lines = []
        for c in i:
            if c in ('', b''):
                raise ValueError('end of stream in command: %r' % line)
            if c in ('.\n', '.', b'.\n', b'.'):
                break
            lines.append(c)
        yield (first, last, lines)