import warnings
import testtools
import fixtures
def test_filters_restored(self):

    class CustomWarning(Warning):
        pass
    fixture = fixtures.WarningsFilter([{'action': 'once', 'category': CustomWarning}])
    old_filters = warnings.filters[:]
    with fixture:
        new_filters = warnings.filters[:]
        self.assertEqual(len(old_filters) + 1, len(new_filters))
        self.assertNotEqual(old_filters, new_filters)
    new_filters = warnings.filters[:]
    self.assertEqual(len(old_filters), len(new_filters))
    self.assertEqual(old_filters, new_filters)