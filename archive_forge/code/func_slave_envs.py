from __future__ import absolute_import
import os
import sys
import socket
import struct
import subprocess
import argparse
import time
import logging
from threading import Thread
def slave_envs(self):
    if self.cmd is None:
        return {}
    else:
        return {'DMLC_PS_ROOT_URI': self.hostIP, 'DMLC_PS_ROOT_PORT': self.port}