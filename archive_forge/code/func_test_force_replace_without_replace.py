import unittest
from charset_normalizer.cli.normalizer import cli_detect, query_yes_no
from unittest.mock import patch
from os.path import exists
from os import remove
def test_force_replace_without_replace(self):
    self.assertEqual(cli_detect(['./data/sample-arabic-1.txt', '--force']), 1)