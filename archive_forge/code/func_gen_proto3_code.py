import argparse
import glob
import os
import re
import subprocess
from textwrap import dedent
from typing import Iterable, Optional
def gen_proto3_code(protoc_path: str, proto3_path: str, include_path: str, cpp_out: str, python_out: str) -> None:
    print(f'Generate pb3 code using {protoc_path}')
    build_args = [protoc_path, proto3_path, '-I', include_path]
    build_args.extend(['--cpp_out', cpp_out, '--python_out', python_out])
    subprocess.check_call(build_args)