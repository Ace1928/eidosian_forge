from __future__ import absolute_import, division, print_function
import argparse
import ast
import os
import re
import requests
import sys
from time import time
import json
def all_ssh_keys(self):
    resp = self.send('account/keys')
    return resp['ssh_keys']