import decimal
import re
from .exceptions import JsonSchemaDefinitionException
from .generator import CodeGenerator, enforce_list
def generate_unique_items(self):
    """
        With Python 3.4 module ``timeit`` recommended this solutions:

        .. code-block:: python

            >>> timeit.timeit("len(x) > len(set(x))", "x=range(100)+range(100)", number=100000)
            0.5839540958404541
            >>> timeit.timeit("len({}.fromkeys(x)) == len(x)", "x=range(100)+range(100)", number=100000)
            0.7094449996948242
            >>> timeit.timeit("seen = set(); any(i in seen or seen.add(i) for i in x)", "x=range(100)+range(100)", number=100000)
            2.0819358825683594
            >>> timeit.timeit("np.unique(x).size == len(x)", "x=range(100)+range(100); import numpy as np", number=100000)
            2.1439831256866455
        """
    unique_definition = self._definition['uniqueItems']
    if not unique_definition:
        return
    self.create_variable_is_list()
    with self.l('if {variable}_is_list:'):
        self.l('def fn(var): return frozenset(dict((k, fn(v)) for k, v in var.items()).items()) if hasattr(var, "items") else tuple(fn(v) for v in var) if isinstance(var, (dict, list)) else str(var) if isinstance(var, bool) else var')
        self.create_variable_with_length()
        with self.l('if {variable}_len > len(set(fn({variable}_x) for {variable}_x in {variable})):'):
            self.exc('{name} must contain unique items', rule='uniqueItems')