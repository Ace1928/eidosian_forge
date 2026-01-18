import contextlib
import os
import signal
import subprocess
import sys
import weakref
import pyarrow as pa
import pytest
def supported_factories():
    yield pa.default_memory_pool
    for backend in pa.supported_memory_backends():
        yield getattr(pa, f'{backend}_memory_pool')