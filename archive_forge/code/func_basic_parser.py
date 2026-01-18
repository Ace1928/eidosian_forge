import argparse
import codecs
import srt
import logging
import sys
import itertools
import os
def basic_parser(description=None, multi_input=False, no_output=False, examples=None, hide_no_strict=False):
    example_lines = []
    if examples is not None:
        example_lines.append('examples:')
        for desc, code in examples.items():
            example_lines.append('  {}'.format(desc))
            example_lines.append('    $ {}\n'.format(code))
    parser = argparse.ArgumentParser(prog=PROG_NAME, description=description, epilog='\n'.join(example_lines), formatter_class=argparse.RawDescriptionHelpFormatter)
    if multi_input:
        parser.add_argument('--input', '-i', metavar='FILE', action='append', type=lambda arg: dash_to_stream(arg, 'input'), help='the files to process', required=True)
    else:
        parser.add_argument('--input', '-i', metavar='FILE', default=STDIN_BYTESTREAM, type=lambda arg: dash_to_stream(arg, 'input'), help='the file to process (default: stdin)')
    if not no_output:
        parser.add_argument('--output', '-o', metavar='FILE', default=STDOUT_BYTESTREAM, type=lambda arg: dash_to_stream(arg, 'output'), help='the file to write to (default: stdout)')
        if not multi_input:
            parser.add_argument('--inplace', '-p', action='store_true', help='modify file in place')
    shelp = 'allow blank lines in output, your media player may explode'
    if hide_no_strict:
        shelp = argparse.SUPPRESS
    parser.add_argument('--no-strict', action='store_false', dest='strict', help=shelp)
    parser.add_argument('--debug', action='store_const', dest='log_level', const=logging.DEBUG, default=logging.INFO, help='enable debug logging')
    parser.add_argument('--ignore-parsing-errors', '-c', action='store_true', help='try to keep going, even if there are parsing errors')
    parser.add_argument('--encoding', '-e', help='the encoding to read/write files in (default: utf8)')
    return parser