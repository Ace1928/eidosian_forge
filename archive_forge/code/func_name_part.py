from __future__ import unicode_literals
from pybtex.plugin import Plugin
from pybtex.textutils import tie_or_space
from pybtex.richtext import Text, nbsp
from pybtex.style.template import together, node
@node
def name_part(children, data, before='', tie=False, abbr=False):
    if abbr:
        children = [child.abbreviate() for child in children]
    parts = together(last_tie=True)[children].format_data(data)
    if not parts:
        return Text()
    if tie:
        return Text(before, parts, tie_or_space(parts, nbsp, ' '))
    else:
        return Text(before, parts)