from __future__ import print_function
import os
import sys
import mxnet as mx
import random
import argparse
import cv2
import time
import traceback
def read_worker(args, q_in, q_out):
    """Function that will be spawned to fetch the image
    from the input queue and put it back to output queue.
    Parameters
    ----------
    args: object
    q_in: queue
    q_out: queue
    """
    while True:
        deq = q_in.get()
        if deq is None:
            break
        i, item = deq
        image_encode(args, i, item, q_out)