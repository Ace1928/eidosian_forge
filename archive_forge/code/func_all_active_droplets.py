from __future__ import absolute_import, division, print_function
import argparse
import ast
import os
import re
import requests
import sys
from time import time
import json
def all_active_droplets(self, tag_name=None):
    if tag_name is not None:
        params = {'tag_name': tag_name}
        resp = self.send('droplets/', params=params)
    else:
        resp = self.send('droplets/')
    return resp['droplets']