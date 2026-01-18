import inspect
import unittest
from traits.observation import expression
from traits.observation._anytrait_filter import anytrait_filter
from traits.observation._dict_item_observer import DictItemObserver
from traits.observation._filtered_trait_observer import FilteredTraitObserver
from traits.observation._list_item_observer import ListItemObserver
from traits.observation._metadata_filter import MetadataFilter
from traits.observation._named_trait_observer import NamedTraitObserver
from traits.observation._set_item_observer import SetItemObserver
from traits.observation._observer_graph import ObserverGraph
def test_compile_expr(self):
    observer1 = 1
    observer2 = 2
    observer3 = 3
    observer4 = 4
    expr1 = create_expression(observer1)
    expr2 = create_expression(observer2)
    expr3 = create_expression(observer3)
    expr4 = create_expression(observer4)
    test_expressions = [expr1, expr1 | expr2, expr1 | expr2 | expr3, expr1.then(expr2), expr1.then(expr2).then(expr3), expr1.then(expr2) | expr3.then(expr4), expr1.list_items(), expr1.dict_items(), expr1.set_items(), expr1.anytrait(notify=False), expr1.anytrait(notify=True)]
    for test_expression in test_expressions:
        with self.subTest(expression=test_expression):
            self.assertEqual(expression.compile_expr(test_expression), test_expression._as_graphs())