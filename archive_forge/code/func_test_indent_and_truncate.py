import io
import json
import yaml
from heatclient.common import format_utils
from heatclient.tests.unit.osc import utils
def test_indent_and_truncate(self):
    self.assertIsNone(format_utils.indent_and_truncate(None))
    self.assertIsNone(format_utils.indent_and_truncate(None, truncate=True))
    self.assertEqual('', format_utils.indent_and_truncate(''))
    self.assertEqual('one', format_utils.indent_and_truncate('one'))
    self.assertIsNone(format_utils.indent_and_truncate(None, spaces=2))
    self.assertEqual('', format_utils.indent_and_truncate('', spaces=2))
    self.assertEqual('  one', format_utils.indent_and_truncate('one', spaces=2))
    self.assertEqual('one\ntwo\nthree\nfour\nfive', format_utils.indent_and_truncate('one\ntwo\nthree\nfour\nfive'))
    self.assertEqual('three\nfour\nfive', format_utils.indent_and_truncate('one\ntwo\nthree\nfour\nfive', truncate=True, truncate_limit=3))
    self.assertEqual('  and so on\n  three\n  four\n  five\n  truncated', format_utils.indent_and_truncate('one\ntwo\nthree\nfour\nfive', spaces=2, truncate=True, truncate_limit=3, truncate_prefix='and so on', truncate_postfix='truncated'))