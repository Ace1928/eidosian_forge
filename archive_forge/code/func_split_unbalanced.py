import difflib
from lxml import etree
from lxml.html import fragment_fromstring
import re
def split_unbalanced(chunks):
    """Return (unbalanced_start, balanced, unbalanced_end), where each is
    a list of text and tag chunks.

    unbalanced_start is a list of all the tags that are opened, but
    not closed in this span.  Similarly, unbalanced_end is a list of
    tags that are closed but were not opened.  Extracting these might
    mean some reordering of the chunks."""
    start = []
    end = []
    tag_stack = []
    balanced = []
    for chunk in chunks:
        if not chunk.startswith('<'):
            balanced.append(chunk)
            continue
        endtag = chunk[1] == '/'
        name = chunk.split()[0].strip('<>/')
        if name in empty_tags:
            balanced.append(chunk)
            continue
        if endtag:
            if tag_stack and tag_stack[-1][0] == name:
                balanced.append(chunk)
                name, pos, tag = tag_stack.pop()
                balanced[pos] = tag
            elif tag_stack:
                start.extend([tag for name, pos, tag in tag_stack])
                tag_stack = []
                end.append(chunk)
            else:
                end.append(chunk)
        else:
            tag_stack.append((name, len(balanced), chunk))
            balanced.append(None)
    start.extend([chunk for name, pos, chunk in tag_stack])
    balanced = [chunk for chunk in balanced if chunk is not None]
    return (start, balanced, end)