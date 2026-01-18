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
def test_class_uniformity(self):
    template_rs = '\nHeatTemplateFormatVersion: \'2012-12-12\'\nResources:\n  secret:\n    Type: OS::Heat::RandomString\n    Properties:\n      length: 66\n      character_classes:\n        - class: lettersdigits\n      character_sequences:\n        - sequence: "*$"\n'
    results = self.run_test(template_rs, 10)
    for char in '$*':
        self.check_stats(self.char_counts(results, char), 1.5, 2)