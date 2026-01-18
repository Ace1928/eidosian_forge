import argparse
import pathlib
import subprocess
import sys
import __main__
def get_device_tests():
    """Returns the location of the device integration tests."""
    return str(pathlib.Path(__file__).parent.absolute())