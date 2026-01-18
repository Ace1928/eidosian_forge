import os
import re
from typing import Iterator, List, Optional
from .errors import BzrError
def iter_file_patch(iter_lines: Iterator[bytes], allow_dirty: bool=False, keep_dirty: bool=False):
    """
    :arg iter_lines: iterable of lines to parse for patches
    :kwarg allow_dirty: If True, allow comments and other non-patch text
        before the first patch.  Note that the algorithm here can only find
        such text before any patches have been found.  Comments after the
        first patch are stripped away in iter_hunks() if it is also passed
        allow_dirty=True.  Default False.
    """
    regex = re.compile(binary_files_re)
    saved_lines: List[bytes] = []
    dirty_head: List[bytes] = []
    orig_range = 0
    beginning = True
    for line in iter_lines:
        if line.startswith(b'=== '):
            if allow_dirty and beginning:
                pass
            elif len(saved_lines) > 0:
                if keep_dirty and len(dirty_head) > 0:
                    yield {'saved_lines': saved_lines, 'dirty_head': dirty_head}
                    dirty_head = []
                else:
                    yield saved_lines
                saved_lines = []
            dirty_head.append(line)
            continue
        if line.startswith(b'*** '):
            continue
        if line.startswith(b'#'):
            continue
        elif orig_range > 0:
            if line.startswith(b'-') or line.startswith(b' '):
                orig_range -= 1
        elif line.startswith(b'--- ') or regex.match(line):
            if allow_dirty and beginning:
                beginning = False
            elif len(saved_lines) > 0:
                if keep_dirty and len(dirty_head) > 0:
                    yield {'saved_lines': saved_lines, 'dirty_head': dirty_head}
                    dirty_head = []
                else:
                    yield saved_lines
            saved_lines = []
        elif line.startswith(b'@@'):
            hunk = hunk_from_header(line)
            orig_range = hunk.orig_range
        saved_lines.append(line)
    if len(saved_lines) > 0:
        if keep_dirty and len(dirty_head) > 0:
            yield {'saved_lines': saved_lines, 'dirty_head': dirty_head}
        else:
            yield saved_lines