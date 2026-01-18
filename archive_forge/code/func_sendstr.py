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
def sendstr(self, s):
    self.sendint(len(s))
    self.sock.sendall(s.encode())