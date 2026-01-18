import platform, subprocess, sys, os
import socket, time
import argparse
def check_hardware():
    print('----------Hardware Info----------')
    print('machine      :', platform.machine())
    print('processor    :', platform.processor())
    if sys.platform.startswith('darwin'):
        pipe = subprocess.Popen(('sysctl', '-a'), stdout=subprocess.PIPE)
        output = pipe.communicate()[0]
        for line in output.split(b'\n'):
            if b'brand_string' in line or b'features' in line:
                print(line.strip())
    elif sys.platform.startswith('linux'):
        subprocess.call(['lscpu'])
    elif sys.platform.startswith('win32'):
        subprocess.call(['wmic', 'cpu', 'get', 'name'])