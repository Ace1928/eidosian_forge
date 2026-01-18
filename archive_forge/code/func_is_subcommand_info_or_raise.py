from abc import ABC
from argparse import ArgumentParser, RawTextHelpFormatter
from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional, Tuple, Type
def is_subcommand_info_or_raise(self):
    if not self.is_subcommand_info:
        raise ValueError(f'The command info must define a subcommand_class attribute, but got: {self}.')