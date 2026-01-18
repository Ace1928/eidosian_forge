import sys
import os
import boto
import optparse
import copy
import boto.exception
import boto.roboto.awsqueryservice
import bdb
import traceback
def add_standard_options(self):
    group = optparse.OptionGroup(self.parser, 'Standard Options')
    group.add_option('-D', '--debug', action='store_true', help='Turn on all debugging output')
    group.add_option('--debugger', action='store_true', default=False, help='Enable interactive debugger on error')
    group.add_option('-U', '--url', action='store', help='Override service URL with value provided')
    group.add_option('--region', action='store', help='Name of the region to connect to')
    group.add_option('-I', '--access-key-id', action='store', help='Override access key value')
    group.add_option('-S', '--secret-key', action='store', help='Override secret key value')
    group.add_option('--version', action='store_true', help='Display version string')
    if self.Filters:
        self.group.add_option('--help-filters', action='store_true', help='Display list of available filters')
        self.group.add_option('--filter', action='append', metavar=' name=value', help='A filter for limiting the results')
    self.parser.add_option_group(group)