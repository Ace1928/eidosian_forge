from abc import ABC
from argparse import ArgumentParser, RawTextHelpFormatter
from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional, Tuple, Type
@property
def is_subcommand_info(self):
    return self.subcommand_class is not None