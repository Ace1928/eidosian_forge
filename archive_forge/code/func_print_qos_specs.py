import sys
import time
import prettytable
from cinderclient import exceptions
from cinderclient import utils
def print_qos_specs(qos_specs):
    print_dict(qos_specs._info, formatters=['specs'])