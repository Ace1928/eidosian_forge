import argparse
import textwrap
from cliff import sphinxext
from cliff.tests import base
def test_metavar(self):
    """Handle an option with a metavar."""
    parser = argparse.ArgumentParser(prog='hello-world', add_help=False)
    parser.add_argument('names', metavar='<NAME>', nargs='+', help='a user name')
    output = '\n'.join(sphinxext._format_parser(parser))
    self.assertEqual(textwrap.dedent('\n        .. program:: hello-world\n        .. code-block:: shell\n\n            hello-world <NAME> [<NAME> ...]\n\n        .. option:: NAME\n\n            a user name\n        ').lstrip(), output)