from __future__ import annotations
import logging # isort:skip
import argparse
import sys
from abc import abstractmethod
from os.path import splitext
from ...document import Document
from ..subcommand import (
from ..util import build_single_handler_applications, die
@classmethod
def other_args(cls) -> Args:
    """ Return args for ``-o`` / ``--output`` to specify where output
        should be written, and for a ``--args`` to pass on any additional
        command line args to the subcommand.

        Subclasses should append these to their class ``args``.

        Example:

            .. code-block:: python

                class Foo(FileOutputSubcommand):

                    args = (

                        FileOutputSubcommand.files_arg("FOO"),

                        # more args for Foo

                    ) + FileOutputSubcommand.other_args()

        """
    return ((('-o', '--output'), Argument(metavar='FILENAME', action='append', type=str, help='Name of the output file or - for standard output.')), ('--args', Argument(metavar='COMMAND-LINE-ARGS', nargs='...', help='Any command line arguments remaining are passed on to the application handler')))