import os
import sys
import asyncio
import threading
from uuid import uuid4
from datetime import datetime, timedelta
from functools import lru_cache, wraps
from lazyops.models import LazyData
from lazyops.common import lazy_import, lazylibs
from lazyops.retry import retryable
def latest_tf_ckpt(model_path):
    ckpt_mtg = lazy_import('tensorflow.python.training.checkpoint_management')
    latest = ckpt_mtg.latest_checkpoint(model_path)
    ckpt_num = int(latest.split('/')[-1].split('-')[-1])
    return LazyData(string=f'Latest Checkpoint = Step {ckpt_num} @ {latest}', value={'step': ckpt_num, 'path': model_path, 'latest_ckpt': latest}, dtype='tfcheckpoint')