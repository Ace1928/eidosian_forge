import platform, subprocess, sys, os
import socket, time
import argparse
def check_os():
    print('----------System Info----------')
    print('Platform     :', platform.platform())
    print('system       :', platform.system())
    print('node         :', platform.node())
    print('release      :', platform.release())
    print('version      :', platform.version())