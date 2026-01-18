import dataclasses
import json
import sys
import types
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser, ArgumentTypeError
from copy import copy
from enum import Enum
from inspect import isclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Literal, NewType, Optional, Tuple, Union, get_type_hints
import yaml
def parse_args_into_dataclasses(self, args=None, return_remaining_strings=False, look_for_args_file=True, args_filename=None, args_file_flag=None) -> Tuple[DataClass, ...]:
    """
        Parse command-line args into instances of the specified dataclass types.

        This relies on argparse's `ArgumentParser.parse_known_args`. See the doc at:
        docs.python.org/3.7/library/argparse.html#argparse.ArgumentParser.parse_args

        Args:
            args:
                List of strings to parse. The default is taken from sys.argv. (same as argparse.ArgumentParser)
            return_remaining_strings:
                If true, also return a list of remaining argument strings.
            look_for_args_file:
                If true, will look for a ".args" file with the same base name as the entry point script for this
                process, and will append its potential content to the command line args.
            args_filename:
                If not None, will uses this file instead of the ".args" file specified in the previous argument.
            args_file_flag:
                If not None, will look for a file in the command-line args specified with this flag. The flag can be
                specified multiple times and precedence is determined by the order (last one wins).

        Returns:
            Tuple consisting of:

                - the dataclass instances in the same order as they were passed to the initializer.abspath
                - if applicable, an additional namespace for more (non-dataclass backed) arguments added to the parser
                  after initialization.
                - The potential list of remaining argument strings. (same as argparse.ArgumentParser.parse_known_args)
        """
    if args_file_flag or args_filename or (look_for_args_file and len(sys.argv)):
        args_files = []
        if args_filename:
            args_files.append(Path(args_filename))
        elif look_for_args_file and len(sys.argv):
            args_files.append(Path(sys.argv[0]).with_suffix('.args'))
        if args_file_flag:
            args_file_parser = ArgumentParser()
            args_file_parser.add_argument(args_file_flag, type=str, action='append')
            cfg, args = args_file_parser.parse_known_args(args=args)
            cmd_args_file_paths = vars(cfg).get(args_file_flag.lstrip('-'), None)
            if cmd_args_file_paths:
                args_files.extend([Path(p) for p in cmd_args_file_paths])
        file_args = []
        for args_file in args_files:
            if args_file.exists():
                file_args += args_file.read_text().split()
        args = file_args + args if args is not None else file_args + sys.argv[1:]
    namespace, remaining_args = self.parse_known_args(args=args)
    outputs = []
    for dtype in self.dataclass_types:
        keys = {f.name for f in dataclasses.fields(dtype) if f.init}
        inputs = {k: v for k, v in vars(namespace).items() if k in keys}
        for k in keys:
            delattr(namespace, k)
        obj = dtype(**inputs)
        outputs.append(obj)
    if len(namespace.__dict__) > 0:
        outputs.append(namespace)
    if return_remaining_strings:
        return (*outputs, remaining_args)
    else:
        if remaining_args:
            raise ValueError(f'Some specified arguments are not used by the HfArgumentParser: {remaining_args}')
        return (*outputs,)