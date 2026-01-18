import os
import os.path
import sys
import glob
import textwrap
import logging
import socket
import subprocess
import pyomo.common
from pyomo.common.collections import Bunch
import pyomo.scripting.pyomo_parser
def print_components(data):
    """
    Print information about modeling components supported by Pyomo.
    """
    from pyomo.core.base.component import ModelComponentFactory, GlobalSets
    print('')
    print('----------------------------------------------------------------')
    print('Pyomo Model Components:')
    print('----------------------------------------------------------------')
    for name in sorted(ModelComponentFactory):
        print('')
        print(' ' + name)
        for line in textwrap.wrap(ModelComponentFactory.doc(name), 59):
            print('    ' + line)
    print('')
    print('----------------------------------------------------------------')
    print('Pyomo Virtual Sets:')
    print('----------------------------------------------------------------')
    for name, obj in sorted(GlobalSets.items()):
        print('')
        print(' ' + name)
        print('    ' + obj.doc)