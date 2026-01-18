from __future__ import annotations
import argparse
import json
import subprocess
import sys
from pathlib import Path
import trio
import trio.testing
def run_pyright(platform: str) -> subprocess.CompletedProcess[bytes]:
    return subprocess.run(['pyright', f'--pythonplatform={platform}', '--pythonversion=3.8', '--verifytypes=trio', '--outputjson', '--ignoreexternal'], capture_output=True)