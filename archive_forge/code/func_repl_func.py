import re
from yaql.language import specs
from yaql.language import yaqltypes
def repl_func(match):
    new_context = context.create_child_context()
    _publish_match(context, match)
    return repl(new_context)