import codecs
import warnings
import re
from contextlib import contextmanager
from parso.normalizer import Normalizer, NormalizerConfig, Issue, Rule
from parso.python.tokenize import _get_token_collection
def process_comp_for(comp_for):
    if comp_for.type == 'sync_comp_for':
        comp = comp_for
    elif comp_for.type == 'comp_for':
        comp = comp_for.children[1]
    exprlist.extend(_get_for_stmt_definition_exprs(comp))