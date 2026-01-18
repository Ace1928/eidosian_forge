import argparse
import os
import platform
import subprocess
import sys
import time
def find_all_gyptest_files(directory):
    result = []
    for root, dirs, files in os.walk(directory):
        result.extend([os.path.join(root, f) for f in files if is_test_name(f)])
    result.sort()
    return result