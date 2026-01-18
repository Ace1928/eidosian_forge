import csv
import logging
import os
import random
import sys
import futurist
from taskflow import engines
from taskflow.patterns import unordered_flow as uf
from taskflow import task
def make_flow(table):
    f = uf.Flow('root')
    for i, row in enumerate(table):
        f.add(RowMultiplier('m-%s' % i, i, row, MULTIPLER))
    return f