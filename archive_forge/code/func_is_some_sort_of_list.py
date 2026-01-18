import contextlib
import re
import textwrap
from typing import Iterable, List, Tuple, Union
def is_some_sort_of_list(text: str, strict: bool, rest_sections: str, style: str) -> bool:
    """Determine if docstring is a reST list.

    Notes
    -----
    There are five types of lists in reST/docutils that need to be handled.

    * `Bullet lists
    <https://docutils.sourceforge.io/docs/user/rst/quickref.html#bullet-lists>`_
    * `Enumerated lists
    <https://docutils.sourceforge.io/docs/user/rst/quickref.html#enumerated-lists>`_
    * `Definition lists
    <https://docutils.sourceforge.io/docs/user/rst/quickref.html#definition-lists>`_
    * `Field lists
    <https://docutils.sourceforge.io/docs/user/rst/quickref.html#field-lists>`_
    * `Option lists
    <https://docutils.sourceforge.io/docs/user/rst/quickref.html#option-lists>`_
    """
    split_lines = text.rstrip().splitlines()
    if len(split_lines) / max([len(line.strip()) for line in split_lines] + [1]) > HEURISTIC_MIN_LIST_ASPECT_RATIO and (not strict):
        return True
    if is_some_sort_of_field_list(text, style):
        return False
    return any((re.match(BULLET_REGEX, line) or re.match(ENUM_REGEX, line) or re.match(rest_sections, line) or re.match(OPTION_REGEX, line) or re.match(EPYTEXT_REGEX, line) or re.match(SPHINX_REGEX, line) or re.match(NUMPY_REGEX, line) or re.match('^\\s*-+', line) or re.match(GOOGLE_REGEX, line) or re.match('[\\S ]+ - \\S+', line) or re.match('\\s*\\S+\\s+--\\s+', line) or re.match(LITERAL_REGEX, line) or re.match('^ *@[a-zA-Z0-9_\\- ]*(?:(?!:).)*$', line) or re.match(' *\\w *:[a-zA-Z0-9_\\- ]*:', line) or re.match(ALEMBIC_REGEX, line) for line in split_lines))