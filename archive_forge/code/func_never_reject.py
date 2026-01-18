import six
import sys
import time
import traceback
import random
import asyncio
import functools
@staticmethod
def never_reject(result):
    return False