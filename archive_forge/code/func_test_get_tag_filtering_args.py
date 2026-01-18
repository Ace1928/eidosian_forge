import argparse
import sys
from unittest import mock
from osc_lib.tests import utils as test_utils
from osc_lib.utils import tags
def test_get_tag_filtering_args(self):
    parser = argparse.ArgumentParser()
    tags.add_tag_filtering_option_to_parser(parser, 'test')
    parsed_args = parser.parse_args(['--tags', 'tag1,tag2', '--any-tags', 'tag4', '--not-tags', 'tag5', '--not-any-tags', 'tag6'])
    expected = {'tags': 'tag1,tag2', 'any_tags': 'tag4', 'not_tags': 'tag5', 'not_any_tags': 'tag6'}
    args = {}
    tags.get_tag_filtering_args(parsed_args, args)
    self.assertEqual(expected, args)