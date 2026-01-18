import re
from unittest import mock
from testtools import matchers
from heat.common import exception
from heat.common import template_format
from heat.engine import node_data
from heat.engine import stack as parser
from heat.engine import template
from heat.tests import common
from heat.tests import utils
def test_generate_random_string_backward_compatible(self):
    stack = self.parse_stack(template_format.parse(self.template_rs))
    secret = stack['secret']
    char_classes = secret.properties['character_classes']
    for char_cl in char_classes:
        char_cl['class'] = self.seq
    for i in range(1, 32):
        r = secret._generate_random_string([], char_classes, self.length)
        self.assertThat(r, matchers.HasLength(self.length))
        regex = '%s{%s}' % (self.pattern, self.length)
        self.assertThat(r, matchers.MatchesRegex(regex))