import argparse
import logging
import pyomo.scripting.pyomo_parser
def setup_test_parser(parser):
    parser.add_argument('--csv-file', '--csv', action='store', dest='csv', default=None, help='Save test results to this file in a CSV format')
    parser.add_argument('-d', '--debug', action='store_true', dest='debug', default=False, help='Show debugging information and text generated during tests.')
    parser.add_argument('-v', '--verbose', action='store_true', dest='verbose', default=False, help='Show verbose results output.')
    parser.add_argument('solver', metavar='SOLVER', default=None, nargs='*', help='a solver name')