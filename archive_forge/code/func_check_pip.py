import platform, subprocess, sys, os
import socket, time
import argparse
def check_pip():
    print('------------Pip Info-----------')
    try:
        import pip
        print('Version      :', pip.__version__)
        print('Directory    :', os.path.dirname(pip.__file__))
    except ImportError:
        print('No corresponding pip install for current python.')