import asyncio
import binascii
from collections import defaultdict
import contextlib
import errno
import functools
import importlib
import inspect
import json
import logging
import multiprocessing
import os
import platform
import re
import signal
import subprocess
import sys
import tempfile
import threading
import time
from urllib.parse import urlencode, unquote, urlparse, parse_qsl, urlunparse
import warnings
from inspect import signature
from pathlib import Path
from subprocess import list2cmdline
from typing import (
import psutil
from google.protobuf import json_format
import ray
import ray._private.ray_constants as ray_constants
from ray.core.generated.runtime_env_common_pb2 import (
def detect_fate_sharing_support_win32():
    global win32_job, win32_AssignProcessToJobObject
    if win32_job is None and sys.platform == 'win32':
        import ctypes
        try:
            from ctypes.wintypes import BOOL, DWORD, HANDLE, LPCWSTR, LPVOID
            kernel32 = ctypes.WinDLL('kernel32')
            kernel32.CreateJobObjectW.argtypes = (LPVOID, LPCWSTR)
            kernel32.CreateJobObjectW.restype = HANDLE
            sijo_argtypes = (HANDLE, ctypes.c_int, LPVOID, DWORD)
            kernel32.SetInformationJobObject.argtypes = sijo_argtypes
            kernel32.SetInformationJobObject.restype = BOOL
            kernel32.AssignProcessToJobObject.argtypes = (HANDLE, HANDLE)
            kernel32.AssignProcessToJobObject.restype = BOOL
            kernel32.IsDebuggerPresent.argtypes = ()
            kernel32.IsDebuggerPresent.restype = BOOL
        except (AttributeError, TypeError, ImportError):
            kernel32 = None
        job = kernel32.CreateJobObjectW(None, None) if kernel32 else None
        job = subprocess.Handle(job) if job else job
        if job:
            from ctypes.wintypes import DWORD, LARGE_INTEGER, ULARGE_INTEGER

            class JOBOBJECT_BASIC_LIMIT_INFORMATION(ctypes.Structure):
                _fields_ = [('PerProcessUserTimeLimit', LARGE_INTEGER), ('PerJobUserTimeLimit', LARGE_INTEGER), ('LimitFlags', DWORD), ('MinimumWorkingSetSize', ctypes.c_size_t), ('MaximumWorkingSetSize', ctypes.c_size_t), ('ActiveProcessLimit', DWORD), ('Affinity', ctypes.c_size_t), ('PriorityClass', DWORD), ('SchedulingClass', DWORD)]

            class IO_COUNTERS(ctypes.Structure):
                _fields_ = [('ReadOperationCount', ULARGE_INTEGER), ('WriteOperationCount', ULARGE_INTEGER), ('OtherOperationCount', ULARGE_INTEGER), ('ReadTransferCount', ULARGE_INTEGER), ('WriteTransferCount', ULARGE_INTEGER), ('OtherTransferCount', ULARGE_INTEGER)]

            class JOBOBJECT_EXTENDED_LIMIT_INFORMATION(ctypes.Structure):
                _fields_ = [('BasicLimitInformation', JOBOBJECT_BASIC_LIMIT_INFORMATION), ('IoInfo', IO_COUNTERS), ('ProcessMemoryLimit', ctypes.c_size_t), ('JobMemoryLimit', ctypes.c_size_t), ('PeakProcessMemoryUsed', ctypes.c_size_t), ('PeakJobMemoryUsed', ctypes.c_size_t)]
            debug = kernel32.IsDebuggerPresent()
            JobObjectExtendedLimitInformation = 9
            JOB_OBJECT_LIMIT_BREAKAWAY_OK = 2048
            JOB_OBJECT_LIMIT_DIE_ON_UNHANDLED_EXCEPTION = 1024
            JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE = 8192
            buf = JOBOBJECT_EXTENDED_LIMIT_INFORMATION()
            buf.BasicLimitInformation.LimitFlags = (0 if debug else JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE) | JOB_OBJECT_LIMIT_DIE_ON_UNHANDLED_EXCEPTION | JOB_OBJECT_LIMIT_BREAKAWAY_OK
            infoclass = JobObjectExtendedLimitInformation
            if not kernel32.SetInformationJobObject(job, infoclass, ctypes.byref(buf), ctypes.sizeof(buf)):
                job = None
        win32_AssignProcessToJobObject = kernel32.AssignProcessToJobObject if kernel32 is not None else False
        win32_job = job if job else False
    return bool(win32_job)