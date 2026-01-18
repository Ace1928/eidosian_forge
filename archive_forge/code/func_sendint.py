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
def sendint(self, n):
    self.sock.sendall(struct.pack('@i', n))