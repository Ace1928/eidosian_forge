import argparse
import sys
from typing import Any
from uuid import UUID
from .main import decode
from .main import encode
from .main import uuid
def encode_cli(args: argparse.Namespace):
    print(encode(args.uuid))