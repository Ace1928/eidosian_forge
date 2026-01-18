from __future__ import annotations
import argparse
import os
import sys
from argparse import ArgumentTypeError
def parse_build_tag(build_tag: str) -> str:
    if build_tag and (not build_tag[0].isdigit()):
        raise ArgumentTypeError('build tag must begin with a digit')
    elif '-' in build_tag:
        raise ArgumentTypeError("invalid character ('-') in build tag")
    return build_tag