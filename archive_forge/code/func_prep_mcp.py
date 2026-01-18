import os
import sys
import json
from os.path import isdir
import subprocess
import pathlib
import setuptools
from setuptools import Command
from setuptools.command.develop import develop
from setuptools.command.install import install
from setuptools.command.install_lib import install_lib
from distutils.command.build import build
from setuptools.dist import Distribution
import shutil
def prep_mcp():
    mydir = os.path.abspath(os.path.dirname(__file__))
    if os.name == 'nt':
        old_dir = os.getcwd()
        os.chdir(os.path.join(mydir, 'scripts'))
        try:
            setup_output = subprocess.check_output(['bash.exe', 'setup_mcp.sh']).decode(errors='ignore')
            if 'ERROR: JAVA_HOME' in setup_output:
                raise RuntimeError('\n                    `java` and/or `javac` commands were not found by the installation script.\n                    Make sure you have installed Java JDK 8.\n                    On Windows, if you installed WSL/WSL2, you may need to install JDK 8 in your WSL\n                    environment with `sudo apt update; sudo apt install openjdk-8-jdk`.\n                    ')
            elif 'Cannot lock task history' in setup_output:
                raise RuntimeError('\n                    Installation failed probably due to Java processes dangling around from previous attempts.\n                    Try killing all Java processes in Windows and WSL (if you use it). Rebooting machine\n                    should also work.\n                    ')
            subprocess.check_call(['bash.exe', 'patch_mcp.sh'])
        except subprocess.CalledProcessError as e:
            raise RuntimeError('\n                Running install scripts failed. Check error logs above for more information.\n\n                If errors are about `bash` command not found, You have at least two options to fix this:\n                 1. Install Windows Subsystem for Linux (WSL. Tested on WSL 2). Note that installation with WSL\n                    may seem especially slow/stuck, but it is not; it is just a bit slow.\n                 2. Install bash along some other tools. E.g., git will come with bash: https://git-scm.com/downloads .\n                    After installation, you may have to update environment variables to include a path which contains\n                    \'bash.exe\'. For above git tools, this is [installation-dir]/bin.\n                After installation, you should have \'bash\' command in your command line/powershell.\n\n                If errors are about "could not create work tree dir...", try cloning the MineRL repository\n                to a different location and try installation again.\n                ')
        os.chdir(old_dir)
    else:
        subprocess.check_call(['bash', os.path.join(mydir, 'scripts', 'setup_mcp.sh')])
        subprocess.check_call(['bash', os.path.join(mydir, 'scripts', 'patch_mcp.sh')])
    gradlew = 'gradlew.bat' if os.name == 'nt' else './gradlew'
    workdir = os.path.join(mydir, 'minerl', 'MCP-Reborn')
    if os.name == 'nt':
        old_dir = os.getcwd()
        os.chdir(workdir)
    n_trials = 3
    for i in range(n_trials):
        try:
            subprocess.check_call('{} downloadAssets'.format(gradlew).split(' '), cwd=workdir)
        except subprocess.CalledProcessError as e:
            if i == n_trials - 1:
                raise e
        else:
            break
    unpack_assets()
    subprocess.check_call('{} clean build shadowJar'.format(gradlew).split(' '), cwd=workdir)
    if os.name == 'nt':
        os.chdir(old_dir)