import six
import sys
import time
import traceback
import random
import asyncio
import functools
@staticmethod
def always_reject(result):
    return True