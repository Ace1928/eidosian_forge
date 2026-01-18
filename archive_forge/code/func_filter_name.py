from parso.tree import search_ancestor
from parso.python.tree import Name
from jedi import settings
from jedi.inference.arguments import TreeArguments
from jedi.inference.value import iterable
from jedi.inference.base_value import NO_VALUES
from jedi.parser_utils import is_scope
def filter_name(filters, name_or_str):
    """
    Searches names that are defined in a scope (the different
    ``filters``), until a name fits.
    """
    string_name = name_or_str.value if isinstance(name_or_str, Name) else name_or_str
    names = []
    for filter in filters:
        names = filter.get(string_name)
        if names:
            break
    return list(_remove_del_stmt(names))