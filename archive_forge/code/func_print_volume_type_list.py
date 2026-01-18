import sys
import time
import prettytable
from cinderclient import exceptions
from cinderclient import utils
def print_volume_type_list(vtypes):
    print_list(vtypes, ['ID', 'Name', 'Description', 'Is_Public'])