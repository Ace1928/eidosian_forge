from __future__ import annotations
import argparse
import os
import sys
from argparse import ArgumentTypeError
def pack_f(args):
    from .pack import pack
    pack(args.directory, args.dest_dir, args.build_number)