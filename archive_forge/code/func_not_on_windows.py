import platform
from pybind11.setup_helpers import Pybind11Extension
from setuptools import Extension
def not_on_windows(s: str) -> str:
    return s if platform.system().lower() != 'windows' else ''