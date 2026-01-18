import collections
import re
import sys
from yaql.language import exceptions
from yaql.language import lexer
def convert_sets_to_lists(engine):
    return engine.options.get('yaql.convertSetsToLists', False)