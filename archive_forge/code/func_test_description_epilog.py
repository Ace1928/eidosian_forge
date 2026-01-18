import argparse
import textwrap
from cliff import sphinxext
from cliff.tests import base
def test_description_epilog(self):
    """Handle a parser description, epilog."""
    parser = argparse.ArgumentParser(prog='hello-world', add_help=False, description='A "Hello, World" app.', epilog='What am I doing down here?')
    parser.add_argument('name', action='store')
    parser.add_argument('--language', dest='lang')
    output = '\n'.join(sphinxext._format_parser(parser))
    self.assertEqual(textwrap.dedent('\n        A "Hello, World" app.\n\n        .. program:: hello-world\n        .. code-block:: shell\n\n            hello-world [--language LANG] name\n\n        .. option:: --language <LANG>\n\n        .. option:: name\n\n        What am I doing down here?\n        ').lstrip(), output)