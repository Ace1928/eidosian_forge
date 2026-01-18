from __future__ import with_statement, print_function
import abc
import sys
from optparse import OptionParser
import rsa
import rsa.pkcs1
def parse_cli(self):
    """Parse the CLI options

        :returns: (cli_opts, cli_args)
        """
    parser = OptionParser(usage=self.usage, description=self.description)
    parser.add_option('-i', '--input', type='string', help=self.input_help)
    if self.has_output:
        parser.add_option('-o', '--output', type='string', help=self.output_help)
    parser.add_option('--keyform', help='Key format of the %s key - default PEM' % self.keyname, choices=('PEM', 'DER'), default='PEM')
    cli, cli_args = parser.parse_args(sys.argv[1:])
    if len(cli_args) != self.expected_cli_args:
        parser.print_help()
        raise SystemExit(1)
    return (cli, cli_args)