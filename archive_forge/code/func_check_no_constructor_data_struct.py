import functools
import re
import tokenize
from hacking import core
@skip_ignored_lines
@core.flake8ext
def check_no_constructor_data_struct(logical_line, filename):
    """Check that data structs (lists, dicts) are declared using literals

    N351
    """
    match = re_no_construct_dict.search(logical_line)
    if match:
        yield (0, 'N351 Remove dict() construct and use literal {}')
    match = re_no_construct_list.search(logical_line)
    if match:
        yield (0, 'N351 Remove list() construct and use literal []')