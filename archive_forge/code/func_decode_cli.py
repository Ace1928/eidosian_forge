import argparse
import sys
from typing import Any
from uuid import UUID
from .main import decode
from .main import encode
from .main import uuid
def decode_cli(args: argparse.Namespace):
    print(str(decode(args.shortuuid, legacy=args.legacy)))