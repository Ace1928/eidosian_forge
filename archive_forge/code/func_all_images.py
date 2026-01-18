from __future__ import absolute_import, division, print_function
import argparse
import ast
import os
import re
import requests
import sys
from time import time
import json
def all_images(self, filter_name='global'):
    params = {'filter': filter_name}
    resp = self.send('images/', data=params)
    return resp['images']