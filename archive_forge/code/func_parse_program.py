from typing import List
from pyquil._parser.parser import run_parser
from pyquil.quil import Program
from pyquil.quilbase import AbstractInstruction
def parse_program(quil: str) -> Program:
    """
    Parse a raw Quil program and return a PyQuil program.

    :param str quil: a single or multiline Quil program
    :return: PyQuil Program object
    """
    return Program(parse(quil))