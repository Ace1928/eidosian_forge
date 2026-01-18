from __future__ import absolute_import
import os
import sys
import socket
import struct
import subprocess
import argparse
import time
import logging
from threading import Thread
def assign_rank(self, rank, wait_conn, tree_map, parent_map, ring_map):
    self.rank = rank
    nnset = set(tree_map[rank])
    rprev, rnext = ring_map[rank]
    self.sock.sendint(rank)
    self.sock.sendint(parent_map[rank])
    self.sock.sendint(len(tree_map))
    self.sock.sendint(len(nnset))
    for r in nnset:
        self.sock.sendint(r)
    if rprev != -1 and rprev != rank:
        nnset.add(rprev)
        self.sock.sendint(rprev)
    else:
        self.sock.sendint(-1)
    if rnext != -1 and rnext != rank:
        nnset.add(rnext)
        self.sock.sendint(rnext)
    else:
        self.sock.sendint(-1)
    while True:
        ngood = self.sock.recvint()
        goodset = set([])
        for _ in range(ngood):
            goodset.add(self.sock.recvint())
        assert goodset.issubset(nnset)
        badset = nnset - goodset
        conset = []
        for r in badset:
            if r in wait_conn:
                conset.append(r)
        self.sock.sendint(len(conset))
        self.sock.sendint(len(badset) - len(conset))
        for r in conset:
            self.sock.sendstr(wait_conn[r].host)
            self.sock.sendint(wait_conn[r].port)
            self.sock.sendint(r)
        nerr = self.sock.recvint()
        if nerr != 0:
            continue
        self.port = self.sock.recvint()
        rmset = []
        for r in conset:
            wait_conn[r].wait_accept -= 1
            if wait_conn[r].wait_accept == 0:
                rmset.append(r)
        for r in rmset:
            wait_conn.pop(r, None)
        self.wait_accept = len(badset) - len(conset)
        return rmset