import sys
import pygame
import pygame.threads
import os
import re
import shutil
import tempfile
import time
import random
from pprint import pformat
def count_results(results):
    total = errors = failures = 0
    for result in results.values():
        if result.get('return_code', 0):
            total += 1
            errors += 1
        else:
            total += result['num_tests']
            errors += result['num_errors']
            failures += result['num_failures']
    return (total, errors, failures)