from abc import abstractmethod
from contextlib import contextmanager
from pathlib import Path
from typing import Optional
from parso.tree import search_ancestor
from parso.python.tree import Name
from jedi.inference.filters import ParserTreeFilter, MergedFilter, \
from jedi.inference.names import AnonymousParamName, TreeNameDefinition
from jedi.inference.base_value import NO_VALUES, ValueSet
from jedi.parser_utils import get_parent_scope
from jedi import debug
from jedi import parser_utils
def py__getattribute__(self, name_or_str, name_context=None, position=None, analysis_errors=True):
    """
        :param position: Position of the last statement -> tuple of line, column
        """
    if name_context is None:
        name_context = self
    names = self.goto(name_or_str, position)
    string_name = name_or_str.value if isinstance(name_or_str, Name) else name_or_str
    found_predefined_types = None
    if self.predefined_names and isinstance(name_or_str, Name):
        node = name_or_str
        while node is not None and (not parser_utils.is_scope(node)):
            node = node.parent
            if node.type in ('if_stmt', 'for_stmt', 'comp_for', 'sync_comp_for'):
                try:
                    name_dict = self.predefined_names[node]
                    types = name_dict[string_name]
                except KeyError:
                    continue
                else:
                    found_predefined_types = types
                    break
    if found_predefined_types is not None and names:
        from jedi.inference import flow_analysis
        check = flow_analysis.reachability_check(context=self, value_scope=self.tree_node, node=name_or_str)
        if check is flow_analysis.UNREACHABLE:
            values = NO_VALUES
        else:
            values = found_predefined_types
    else:
        values = ValueSet.from_sets((name.infer() for name in names))
    if not names and (not values) and analysis_errors:
        if isinstance(name_or_str, Name):
            from jedi.inference import analysis
            message = "NameError: name '%s' is not defined." % string_name
            analysis.add(name_context, 'name-error', name_or_str, message)
    debug.dbg('context.names_to_types: %s -> %s', names, values)
    if values:
        return values
    return self._check_for_additional_knowledge(name_or_str, name_context, position)