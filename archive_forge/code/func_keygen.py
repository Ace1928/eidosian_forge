from __future__ import with_statement, print_function
import abc
import sys
from optparse import OptionParser
import rsa
import rsa.pkcs1
def keygen():
    """Key generator."""
    parser = OptionParser(usage='usage: %prog [options] keysize', description='Generates a new RSA keypair of "keysize" bits.')
    parser.add_option('--pubout', type='string', help='Output filename for the public key. The public key is not saved if this option is not present. You can use pyrsa-priv2pub to create the public key file later.')
    parser.add_option('-o', '--out', type='string', help='Output filename for the private key. The key is written to stdout if this option is not present.')
    parser.add_option('--form', help='key format of the private and public keys - default PEM', choices=('PEM', 'DER'), default='PEM')
    cli, cli_args = parser.parse_args(sys.argv[1:])
    if len(cli_args) != 1:
        parser.print_help()
        raise SystemExit(1)
    try:
        keysize = int(cli_args[0])
    except ValueError:
        parser.print_help()
        print('Not a valid number: %s' % cli_args[0], file=sys.stderr)
        raise SystemExit(1)
    print('Generating %i-bit key' % keysize, file=sys.stderr)
    pub_key, priv_key = rsa.newkeys(keysize)
    if cli.pubout:
        print('Writing public key to %s' % cli.pubout, file=sys.stderr)
        data = pub_key.save_pkcs1(format=cli.form)
        with open(cli.pubout, 'wb') as outfile:
            outfile.write(data)
    data = priv_key.save_pkcs1(format=cli.form)
    if cli.out:
        print('Writing private key to %s' % cli.out, file=sys.stderr)
        with open(cli.out, 'wb') as outfile:
            outfile.write(data)
    else:
        print('Writing private key to stdout', file=sys.stderr)
        rsa._compat.write_to_stdout(data)