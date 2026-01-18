from __future__ import annotations
import os
import argparse
import subprocess
import typing as T
def read_linguas(src_sub: str) -> T.List[str]:
    linguas = os.path.join(src_sub, 'LINGUAS')
    try:
        langs = []
        with open(linguas, encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line and (not line.startswith('#')):
                    langs += line.split()
        return langs
    except (FileNotFoundError, PermissionError):
        print(f'Could not find file LINGUAS in {src_sub}')
        return []