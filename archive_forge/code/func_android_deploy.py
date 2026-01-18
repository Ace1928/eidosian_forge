import importlib
import os
import subprocess
import sys
import sysconfig
from pathlib import Path
import PySide6 as ref_mod
def android_deploy():
    if not sys.platform == 'linux':
        print('pyside6-android-deploy only works from a Linux host')
    else:
        android_requirements_file = Path(__file__).parent / 'requirements-android.txt'
        with open(android_requirements_file, 'r', encoding='UTF-8') as file:
            while (line := file.readline()):
                dependent_package = line.rstrip()
                if not bool(importlib.util.find_spec(dependent_package)):
                    command = [sys.executable, '-m', 'pip', 'install', dependent_package]
                    subprocess.run(command)
        pyside_script_wrapper('android_deploy.py')