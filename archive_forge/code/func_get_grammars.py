import ast
import sys
import warnings
from typing import Iterable, Iterator, List, Set, Tuple
from black.mode import VERSION_TO_FEATURES, Feature, TargetVersion, supports_feature
from black.nodes import syms
from blib2to3 import pygram
from blib2to3.pgen2 import driver
from blib2to3.pgen2.grammar import Grammar
from blib2to3.pgen2.parse import ParseError
from blib2to3.pgen2.tokenize import TokenError
from blib2to3.pytree import Leaf, Node
def get_grammars(target_versions: Set[TargetVersion]) -> List[Grammar]:
    if not target_versions:
        return [pygram.python_grammar_async_keywords, pygram.python_grammar, pygram.python_grammar_soft_keywords]
    grammars = []
    if not supports_feature(target_versions, Feature.ASYNC_IDENTIFIERS) and (not supports_feature(target_versions, Feature.PATTERN_MATCHING)):
        grammars.append(pygram.python_grammar_async_keywords)
    if not supports_feature(target_versions, Feature.ASYNC_KEYWORDS):
        grammars.append(pygram.python_grammar)
    if any((Feature.PATTERN_MATCHING in VERSION_TO_FEATURES[v] for v in target_versions)):
        grammars.append(pygram.python_grammar_soft_keywords)
    return grammars