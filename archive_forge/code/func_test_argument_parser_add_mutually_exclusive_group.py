import unittest
from cliff import _argparse
def test_argument_parser_add_mutually_exclusive_group(self):
    parser = _argparse.ArgumentParser(conflict_handler='ignore')
    parser.add_mutually_exclusive_group()