import os, sys, time, random
def decodeBytes(self, tokens):
    return b''.join(map(lambda i: self.idx2token[i], tokens))