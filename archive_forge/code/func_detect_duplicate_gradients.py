from __future__ import division         # use "true" division instead of integer division in Python 2 (see PEP 238)
from __future__ import print_function   # use print() as a function in Python 2 (see PEP 3105)
from __future__ import absolute_import  # use absolute imports by default in Python 2 (see PEP 328)
import math
import optparse
import os
import re
import sys
import time
import xml.dom.minidom
from xml.dom import Node, NotFoundErr
from collections import namedtuple, defaultdict
from decimal import Context, Decimal, InvalidOperation, getcontext
import six
from six.moves import range, urllib
from scour.svg_regex import svg_parser
from scour.svg_transform import svg_transform_parser
from scour.yocto_css import parseCssString
from scour import __version__
def detect_duplicate_gradients(*grad_lists):
    """Detects duplicate gradients from each iterable/generator given as argument

    Yields (master, master_id, duplicates_id, duplicates) tuples where:
      * master_id: The ID attribute of the master element.  This will always be non-empty
        and not None as long at least one of the gradients have a valid ID.
      * duplicates_id: List of ID attributes of the duplicate gradients elements (can be
        empty where the gradient had no ID attribute)
      * duplicates: List of elements that are duplicates of the `master` element.  Will
        never include the `master` element.  Has the same order as `duplicates_id` - i.e.
        `duplicates[X].getAttribute("id") == duplicates_id[X]`.
    """
    for grads in grad_lists:
        grad_buckets = defaultdict(list)
        for grad in grads:
            key = computeGradientBucketKey(grad)
            grad_buckets[key].append(grad)
        for bucket in six.itervalues(grad_buckets):
            if len(bucket) < 2:
                continue
            master = bucket[0]
            duplicates = bucket[1:]
            duplicates_ids = [d.getAttribute('id') for d in duplicates]
            master_id = master.getAttribute('id')
            if not master_id:
                for i in range(len(duplicates_ids)):
                    dup_id = duplicates_ids[i]
                    if dup_id:
                        master_id = duplicates_ids[i]
                        duplicates[i] = master
                        duplicates_ids[i] = ''
                        break
            yield (master_id, duplicates_ids, duplicates)