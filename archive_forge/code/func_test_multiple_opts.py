import argparse
import textwrap
from cliff import sphinxext
from cliff.tests import base
def test_multiple_opts(self):
    """Correctly output multiple opts on separate lines."""
    parser = argparse.ArgumentParser(prog='hello-world', add_help=False)
    parser.add_argument('name', help='user name')
    parser.add_argument('--language', dest='lang', help='greeting language')
    parser.add_argument('--translate', action='store_true', help='translate to local language')
    parser.add_argument('--write-to-var-log-something-or-other', action='store_true', help='a long opt to force wrapping')
    parser.add_argument('--required-arg', dest='stuff', required=True, help='a required argument')
    style_group = parser.add_mutually_exclusive_group(required=True)
    style_group.add_argument('--polite', action='store_true', help='use a polite greeting')
    style_group.add_argument('--profane', action='store_true', help='use a less polite greeting')
    output = '\n'.join(sphinxext._format_parser(parser))
    self.assertEqual(textwrap.dedent('\n        .. program:: hello-world\n        .. code-block:: shell\n\n            hello-world\n                [--language LANG]\n                [--translate]\n                [--write-to-var-log-something-or-other]\n                --required-arg STUFF\n                (--polite | --profane)\n                name\n\n        .. option:: --language <LANG>\n\n            greeting language\n\n        .. option:: --translate\n\n            translate to local language\n\n        .. option:: --write-to-var-log-something-or-other\n\n            a long opt to force wrapping\n\n        .. option:: --required-arg <STUFF>\n\n            a required argument\n\n        .. option:: --polite\n\n            use a polite greeting\n\n        .. option:: --profane\n\n            use a less polite greeting\n\n        .. option:: name\n\n            user name\n        ').lstrip(), output)