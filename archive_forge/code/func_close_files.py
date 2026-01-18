import argparse
import os
from time import time
from pyzstd import compress_stream, decompress_stream, \
def close_files(args):
    if args.input is not None:
        args.input.close()
    if args.output is not None:
        args.output.close()