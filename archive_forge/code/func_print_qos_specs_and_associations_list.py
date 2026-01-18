import sys
import time
import prettytable
from cinderclient import exceptions
from cinderclient import utils
def print_qos_specs_and_associations_list(q_specs):
    print_list(q_specs, ['ID', 'Name', 'Consumer', 'specs'])