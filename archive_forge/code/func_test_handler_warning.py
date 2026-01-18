import unittest
import warnings
from traits.api import (
def test_handler_warning(self):
    handlers = {'TraitDict': TraitDict, 'TraitList': TraitList, 'TraitTuple': TraitTuple, 'TraitPrefixList': lambda: TraitPrefixList('one', 'two'), 'TraitPrefixMap': lambda: TraitPrefixMap({})}
    for name, handler_factory in handlers.items():
        with self.subTest(handler=name):
            with warnings.catch_warnings(record=True):
                warnings.simplefilter('error', DeprecationWarning)
                with self.assertRaises(DeprecationWarning) as cm:
                    handler_factory()
            self.assertIn(name, str(cm.exception))