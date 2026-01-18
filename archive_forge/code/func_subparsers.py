from abc import ABC
from argparse import ArgumentParser, RawTextHelpFormatter
from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional, Tuple, Type
@subparsers.setter
def subparsers(self, subparsers: Optional['_SubParsersAction']):
    self._subparsers = subparsers