import itertools
from heat.scaling import template
from heat.tests import common
def test_create_template(self):
    """Test case for creating template.

        When creating a template from scratch, an empty list is accepted as
        the "old" resources and new resources are created up to num_resource.
        """
    templates = template.member_definitions([], {'type': 'Foo'}, 2, 0, self.next_id)
    expected = [('stubbed-id-0', {'type': 'Foo'}), ('stubbed-id-1', {'type': 'Foo'})]
    self.assertEqual(expected, list(templates))