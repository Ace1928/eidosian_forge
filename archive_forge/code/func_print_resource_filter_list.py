import sys
import time
import prettytable
from cinderclient import exceptions
from cinderclient import utils
def print_resource_filter_list(filters):
    formatter = {'Filters': lambda resource: ', '.join(resource.filters)}
    print_list(filters, ['Resource', 'Filters'], formatters=formatter)