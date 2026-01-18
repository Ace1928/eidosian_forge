import binascii
import lzma
import platform
import sys
def is_pypy369later() -> bool:
    """check if running platform is PyPY and python 3.6.9 and later"""
    return platform.python_implementation() == 'PyPy' and sys.version_info >= (3, 6, 9)