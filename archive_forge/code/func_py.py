import os
import subprocess
import re
import argparse
from time import strftime
import shutil
from pathlib import Path
def py(cmd, output=False):
    return runcmd('python3 {}'.format(cmd))