from __future__ import absolute_import, division, print_function
import argparse
import ast
import os
import re
import requests
import sys
from time import time
import json
def add_inventory_group(self, key):
    """Method to create group dict"""
    host_dict = {'hosts': [], 'vars': {}}
    self.inventory[key] = host_dict
    return