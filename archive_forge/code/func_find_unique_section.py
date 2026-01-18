import re
from . import utilities
def find_unique_section(text, name):
    """
    Like find_section but ensures that there is only a single
    such section. Exceptions in all other cases.

    >>> t = "abc abc\\n==FOO=BEGINS==\\nbar bar\\n==FOO=ENDS==\\n"
    >>> find_unique_section(t, "FOO")
    'bar bar'
    """
    sections = find_section(text, name)
    if len(sections) > 1:
        raise Exception('Section %s more than once in file' % name)
    if not sections:
        raise Exception('No section %s in file' % name)
    return sections[0]