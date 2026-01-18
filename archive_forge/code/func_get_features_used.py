import io
import json
import platform
import re
import sys
import tokenize
import traceback
from contextlib import contextmanager
from dataclasses import replace
from datetime import datetime, timezone
from enum import Enum
from json.decoder import JSONDecodeError
from pathlib import Path
from typing import (
import click
from click.core import ParameterSource
from mypy_extensions import mypyc_attr
from pathspec import PathSpec
from pathspec.patterns.gitwildmatch import GitWildMatchPatternError
from _black_version import version as __version__
from black.cache import Cache
from black.comments import normalize_fmt_off
from black.const import (
from black.files import (
from black.handle_ipynb_magics import (
from black.linegen import LN, LineGenerator, transform_line
from black.lines import EmptyLineTracker, LinesBlock
from black.mode import FUTURE_FLAG_TO_FEATURE, VERSION_TO_FEATURES, Feature
from black.mode import Mode as Mode  # re-exported
from black.mode import Preview, TargetVersion, supports_feature
from black.nodes import (
from black.output import color_diff, diff, dump_to_file, err, ipynb_diff, out
from black.parsing import (  # noqa F401
from black.ranges import (
from black.report import Changed, NothingChanged, Report
from black.trans import iter_fexpr_spans
from blib2to3.pgen2 import token
from blib2to3.pytree import Leaf, Node
def get_features_used(node: Node, *, future_imports: Optional[Set[str]]=None) -> Set[Feature]:
    """Return a set of (relatively) new Python features used in this file.

    Currently looking for:
    - f-strings;
    - self-documenting expressions in f-strings (f"{x=}");
    - underscores in numeric literals;
    - trailing commas after * or ** in function signatures and calls;
    - positional only arguments in function signatures and lambdas;
    - assignment expression;
    - relaxed decorator syntax;
    - usage of __future__ flags (annotations);
    - print / exec statements;
    - parenthesized context managers;
    - match statements;
    - except* clause;
    - variadic generics;
    """
    features: Set[Feature] = set()
    if future_imports:
        features |= {FUTURE_FLAG_TO_FEATURE[future_import] for future_import in future_imports if future_import in FUTURE_FLAG_TO_FEATURE}
    for n in node.pre_order():
        if is_string_token(n):
            value_head = n.value[:2]
            if value_head in {'f"', 'F"', "f'", "F'", 'rf', 'fr', 'RF', 'FR'}:
                features.add(Feature.F_STRINGS)
                if Feature.DEBUG_F_STRINGS not in features:
                    for span_beg, span_end in iter_fexpr_spans(n.value):
                        if n.value[span_beg:span_end - 1].rstrip().endswith('='):
                            features.add(Feature.DEBUG_F_STRINGS)
                            break
        elif is_number_token(n):
            if '_' in n.value:
                features.add(Feature.NUMERIC_UNDERSCORES)
        elif n.type == token.SLASH:
            if n.parent and n.parent.type in {syms.typedargslist, syms.arglist, syms.varargslist}:
                features.add(Feature.POS_ONLY_ARGUMENTS)
        elif n.type == token.COLONEQUAL:
            features.add(Feature.ASSIGNMENT_EXPRESSIONS)
        elif n.type == syms.decorator:
            if len(n.children) > 1 and (not is_simple_decorator_expression(n.children[1])):
                features.add(Feature.RELAXED_DECORATORS)
        elif n.type in {syms.typedargslist, syms.arglist} and n.children and (n.children[-1].type == token.COMMA):
            if n.type == syms.typedargslist:
                feature = Feature.TRAILING_COMMA_IN_DEF
            else:
                feature = Feature.TRAILING_COMMA_IN_CALL
            for ch in n.children:
                if ch.type in STARS:
                    features.add(feature)
                if ch.type == syms.argument:
                    for argch in ch.children:
                        if argch.type in STARS:
                            features.add(feature)
        elif n.type in {syms.return_stmt, syms.yield_expr} and len(n.children) >= 2 and (n.children[1].type == syms.testlist_star_expr) and any((child.type == syms.star_expr for child in n.children[1].children)):
            features.add(Feature.UNPACKING_ON_FLOW)
        elif n.type == syms.annassign and len(n.children) >= 4 and (n.children[3].type == syms.testlist_star_expr):
            features.add(Feature.ANN_ASSIGN_EXTENDED_RHS)
        elif n.type == syms.with_stmt and len(n.children) > 2 and (n.children[1].type == syms.atom):
            atom_children = n.children[1].children
            if len(atom_children) == 3 and atom_children[0].type == token.LPAR and _contains_asexpr(atom_children[1]) and (atom_children[2].type == token.RPAR):
                features.add(Feature.PARENTHESIZED_CONTEXT_MANAGERS)
        elif n.type == syms.match_stmt:
            features.add(Feature.PATTERN_MATCHING)
        elif n.type == syms.except_clause and len(n.children) >= 2 and (n.children[1].type == token.STAR):
            features.add(Feature.EXCEPT_STAR)
        elif n.type in {syms.subscriptlist, syms.trailer} and any((child.type == syms.star_expr for child in n.children)):
            features.add(Feature.VARIADIC_GENERICS)
        elif n.type == syms.tname_star and len(n.children) == 3 and (n.children[2].type == syms.star_expr):
            features.add(Feature.VARIADIC_GENERICS)
        elif n.type in (syms.type_stmt, syms.typeparams):
            features.add(Feature.TYPE_PARAMS)
    return features