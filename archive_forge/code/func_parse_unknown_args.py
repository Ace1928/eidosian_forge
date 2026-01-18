from argparse import ArgumentParser
from datasets.commands.convert import ConvertCommand
from datasets.commands.dummy_data import DummyDataCommand
from datasets.commands.env import EnvironmentCommand
from datasets.commands.run_beam import RunBeamCommand
from datasets.commands.test import TestCommand
from datasets.utils.logging import set_verbosity_info
def parse_unknown_args(unknown_args):
    return {key.lstrip('-'): value for key, value in zip(unknown_args[::2], unknown_args[1::2])}