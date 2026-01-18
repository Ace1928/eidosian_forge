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
def recvstr(self):
    slen = self.recvint()
    return self.recvall(slen).decode()