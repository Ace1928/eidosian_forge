import argparse
import codecs
import srt
import logging
import sys
import itertools
import os
def dash_to_stream(arg, arg_type):
    if arg == '-':
        return DASH_STREAM_MAP[arg_type]
    return arg