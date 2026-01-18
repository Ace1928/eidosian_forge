import itertools
from heat.scaling import template
from heat.tests import common
def test_growth_counts_as_replacement(self):
    """Test case for growing template.

        If we grow the template and replace some elements at the same time, the
        number of replacements to perform is reduced by the number of new
        resources to be created.
        """
    spec = {'type': 'Foo'}
    old_resources = [('old-id-0', spec), ('old-id-1', spec)]
    new_spec = {'type': 'Bar'}
    templates = template.member_definitions(old_resources, new_spec, 4, 2, self.next_id)
    expected = [('old-id-0', spec), ('old-id-1', spec), ('stubbed-id-0', new_spec), ('stubbed-id-1', new_spec)]
    self.assertEqual(expected, list(templates))