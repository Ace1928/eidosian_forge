from __future__ import division
import datetime
import math
def format_updatable(updatable, pbar):
    if hasattr(updatable, 'update'):
        return updatable.update(pbar)
    else:
        return updatable