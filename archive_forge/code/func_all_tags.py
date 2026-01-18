from __future__ import absolute_import, division, print_function
import argparse
import ast
import os
import re
import requests
import sys
from time import time
import json
def all_tags(self):
    resp = self.send('tags')
    return resp['tags']